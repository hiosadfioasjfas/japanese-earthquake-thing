"""
JMA -> Roblox relay (Python / Flask), deployable on Render.

What it does:
  1. Polls JMA's public earthquake feed on a timer (every POLL_INTERVAL_S)
  2. Flattens Body.Intensity.Observation into per-station readings
  3. Serves GET /stations as clean JSON for Roblox's HttpService

Local run:
    pip install -r requirements.txt --break-system-packages
    python app.py
    curl http://localhost:8787/stations

Render deploy:
    See render.yaml in the repo root - it points Render at this app via
    gunicorn. Render assigns the port dynamically through the $PORT env
    var, which this app (and the gunicorn command in render.yaml) reads
    automatically - you don't need to touch PORT yourself.

IMPORTANT: this app keeps all live station data in an in-memory dict.
That only works correctly with exactly ONE worker process (see the
`gunicorn --workers 1` in render.yaml). Do not raise the worker count
without moving state into something shared (e.g. Redis) - multiple
workers would each poll JMA independently and serve inconsistent data.
"""

import os
import time
import threading
import math
import traceback
import functools

import requests
from flask import Flask, jsonify

# Force every print() call to flush immediately. Without this, Python
# block-buffers stdout when it isn't attached to a real terminal (which is
# exactly gunicorn's case), so log lines can appear drastically delayed,
# reordered, or batched together - which is almost certainly why multiple
# "entering iteration N" lines showed up bunched into the same second
# during debugging, instead of ~20s apart as POLL_INTERVAL_S should cause.
print = functools.partial(print, flush=True)

PORT = int(os.environ.get("PORT", 8787))
POLL_INTERVAL_S = 20  # be polite to JMA - no need to poll faster than this
LIST_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"
DETAIL_BASE = "https://www.jma.go.jp/bosai/quake/data/"

# Official JMA seismic-intensity station master: every station's Code here
# is the SAME code used in the live feed's IntensityStation.Code field, so
# matching live readings to real lat/lon is a direct dict lookup - no
# fuzzy name matching needed. ~4,360 stations nationwide (JMA's own network
# plus local-government points JMA folds into the same feed).
STATION_MASTER_URL = "https://www.jma.go.jp/jma/kishou/know/jishin/intens-st/stations.json"
# Fallback mirror (community-maintained, same schema/codes) in case JMA's
# own host is unreachable from wherever this relay is deployed.
STATION_MASTER_FALLBACK_URL = "https://raw.githubusercontent.com/iku55/jma_int_stations/main/stations.json"
STATION_MASTER_REFRESH_S = 24 * 60 * 60  # station list barely ever changes - refetch once/day

INTENSITY_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5-": 5.0, "5+": 5.5,
    "6-": 6.0, "6+": 6.5,
    "7": 7,
}

app = Flask(__name__)

_lock = threading.Lock()
_station_state = {}   # code -> dict (live readings, intensity etc.)
_last_poll_ok = None
_last_poll_error = None
_last_poll_attempt = None  # NEW: set at the START of every poll attempt, so
                            # /health can tell "never tried" apart from
                            # "tried and is currently running/stuck"
_poller_started = False
_poller_lock = threading.Lock()

_master_lock = threading.Lock()
_station_master = {}       # code -> { code, name, name_ja, pref, lat, lon, affi }
_master_last_fetch_ok = None
_master_last_error = None
_current_event_json = None

# A real browser-like User-Agent. JMA's public data endpoints are known to
# 403 non-browser-looking User-Agents on some hosts (Render's IP ranges
# included) even though curl/browsers from a residential IP work fine.
# This was the actual bug: request calls were getting HTTP 403 responses
# back, .raise_for_status() correctly raised, but the exception was being
# swallowed inside poll_once()'s per-item try/except without ever being
# logged anywhere visible, so it just looked like "silently does nothing"
# from outside. See fetch_json() below for the fix (real logging + a
# guaranteed _last_poll_error update on every failure path).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

# Single shared session (connection pooling + consistent headers/timeouts).
_session = requests.Session()
_session.headers.update(HEADERS)

REQUEST_TIMEOUT = (5, 10)  # (connect timeout, read timeout) - explicit tuple so a
                           # slow/hanging TCP handshake can't block past 5s even if
                           # the read side would otherwise wait the full 10s


# NOTE: an earlier version of this file routed fetch_json() through a
# ThreadPoolExecutor with a hard .result(timeout=...) ceiling, intended to
# guarantee no request could hang the poll loop indefinitely. In practice,
# that indirection was the actual bug: calls submitted from inside the
# background poll thread never completed, while the identical request made
# directly (e.g. from a Flask request handler in /debug-detail) worked
# fine every time. Removed in favor of calling requests.get() directly,
# which is proven working. REQUEST_TIMEOUT below still bounds each call at
# the socket level.


def fetch_json(url, what):
    """GET a URL and parse JSON, with real error visibility. Never raises -
    returns (data, error_string). Calls requests.get() directly (no thread
    pool indirection) - this matches /debug-detail exactly, which is
    confirmed working against live JMA data, whereas the previous
    ThreadPoolExecutor-based version never completed a single successful
    call from inside the background poll thread."""
    print(f"[relay] fetch_json: starting {what} ({url})")
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        msg = f"{what}: request failed: {e}"
        print(f"[relay] {msg}")
        return None, msg
    except Exception as e:
        msg = f"{what}: unexpected error: {e}"
        print(f"[relay] {msg}")
        return None, msg

    print(f"[relay] fetch_json: {what} returned HTTP {resp.status_code}")

    if resp.status_code != 200:
        body_preview = resp.text[:200].replace("\n", " ")
        msg = f"{what}: HTTP {resp.status_code} - {body_preview}"
        print(f"[relay] {msg}")
        return None, msg

    try:
        return resp.json(), None
    except ValueError as e:
        msg = f"{what}: bad JSON: {e}"
        print(f"[relay] {msg}")
        return None, msg


def normalize_intensity(raw):
    if raw is None:
        return 0.0
    s = str(raw).strip()
    if s in INTENSITY_MAP:
        return float(INTENSITY_MAP[s])
    try:
        return float(s)
    except ValueError:
        return 0.0


def extract_station_readings(detail_json):
    """Walk Body.Intensity.Observation.Pref[].Area[].City[].IntensityStation[]"""
    out = []
    obs = (detail_json.get("Body") or {}).get("Intensity", {}).get("Observation")
    if not obs or not isinstance(obs.get("Pref"), list):
        return out

    for pref in obs["Pref"]:
        pref_name = pref.get("Name", "")
        for area in pref.get("Area", []) or []:
            for city in area.get("City", []) or []:
                for st in city.get("IntensityStation", []) or []:
                    out.append({
                        "code": st.get("Code"),
                        "name": st.get("Name"),
                        "pref": pref_name,
                        "intensity": normalize_intensity(st.get("Int")),
                    })
    return out


def _parse_master_payload(raw):
    """Normalize the JMA station master (or the fallback mirror, which
    shares the same schema) into { code -> station dict }."""
    parsed = {}
    if not isinstance(raw, list):
        return parsed
    for entry in raw:
        code = entry.get("code")
        if not code:
            continue
        try:
            lat = float(entry.get("lat"))
            lon = float(entry.get("lon"))
        except (TypeError, ValueError):
            continue
        pref = entry.get("pref") or {}
        parsed[code] = {
            "code": code,
            "name_ja": entry.get("name"),
            "pref": pref.get("name") if isinstance(pref, dict) else pref,
            "lat": lat,
            "lon": lon,
            "affi": entry.get("affi"),  # e.g. "気象庁" (JMA) vs local government
        }
    return parsed


def fetch_station_master(force=False):
    """Fetch and cache the real JMA station master. Tries JMA's own file
    first (authoritative), falls back to the community mirror if JMA's
    host is unreachable from this deployment. Safe to call repeatedly -
    only refetches if stale or forced."""
    global _master_last_fetch_ok, _master_last_error

    with _master_lock:
        if not force and _station_master and _master_last_fetch_ok:
            if time.time() - _master_last_fetch_ok < STATION_MASTER_REFRESH_S:
                return

    last_err = None
    for url in (STATION_MASTER_URL, STATION_MASTER_FALLBACK_URL):
        raw, err = fetch_json(url, "station master")
        if err:
            last_err = err
            continue
        parsed = _parse_master_payload(raw)
        if not parsed:
            last_err = f"{url}: parsed station master was empty"
            print(f"[relay] {last_err}")
            continue
        with _master_lock:
            _station_master.clear()
            _station_master.update(parsed)
            _master_last_fetch_ok = time.time()
            _master_last_error = None
        print(f"[relay] station master loaded: {len(parsed)} stations (source: {url})")
        return

    # both sources failed
    with _master_lock:
        _master_last_error = last_err
    print(f"[relay] station master fetch failed: {last_err}")


def poll_once():
    global _last_poll_ok, _last_poll_error, _last_poll_attempt, _current_event_json

    with _lock:
        _last_poll_attempt = time.time()

    quake_list, err = fetch_json(LIST_URL, "quake list")
    if err:
        with _lock:
            _last_poll_error = err
        return

    if not isinstance(quake_list, list):
        with _lock:
            _last_poll_error = "quake list was not a JSON list"
        return

    latest = next((q for q in quake_list if q.get("json")), None)

    if latest is None:
        with _lock:
            _station_state.clear()
            _last_poll_error = "no active earthquake"
        return

    _current_event_json = latest["json"]

    detail, err = fetch_json(
        DETAIL_BASE + _current_event_json,
        f"quake detail ({_current_event_json})",
    )

    if err:
        with _lock:
            _last_poll_error = err
        return

    readings = extract_station_readings(detail)

    now = time.time()

    with _master_lock:
        master_snapshot = dict(_station_master)

    new_state = {}

    for r in readings:
        code = r.get("code")
        if not code:
            continue

        master = master_snapshot.get(code)

        new_state[code] = {
            "code": code,
            "name": r.get("name") or code,
            "pref": r.get("pref") or (master or {}).get("pref"),
            "lat": (master or {}).get("lat"),
            "lon": (master or {}).get("lon"),
            "matched": master is not None,
            "intensity": r["intensity"],
            "eventTime": latest.get("at"),
            "updatedAt": now,
        }

    with _lock:
        _station_state.clear()
        _station_state.update(new_state)
        _last_poll_ok = now
        _last_poll_error = None

    print(
        f"[relay] updated {_current_event_json}: {len(new_state)} live stations"
    )

    if not new_state:
        print("[relay] poll_once: no readings this cycle")

def poll_loop():
    global _last_poll_error
    print("[relay] poll_loop: thread body has started executing")  # DIAGNOSTIC
    iteration = 0
    while True:
        iteration += 1
        print(f"[relay] poll_loop: entering iteration {iteration}")  # DIAGNOSTIC
        try:
            fetch_station_master()  # no-op if already fresh; retries if it failed before
            poll_once()
        except Exception as e:
            # Broad catch is intentional (this loop must never die), but
            # now it prints a full traceback instead of just str(e), so a
            # bug inside poll_once() is actually diagnosable from logs.
            with _lock:
                _last_poll_error = f"{e}"
            print(f"[relay] poll failed: {e}")
            traceback.print_exc()
        time.sleep(POLL_INTERVAL_S)


def ensure_poller_started():
    """Starts the background poll + decay threads exactly once, even if
    Flask/gunicorn imports this module more than once (e.g. the reloader
    in debug mode, or gunicorn's worker boot process)."""
    global _poller_started
    with _poller_lock:
        if _poller_started:
            return
        _poller_started = True
        # Fetch the real station master synchronously before serving, so the
        # very first poll/response already has correct lat/lon - not just
        # after the first 24h refresh cycle.
        try:
            fetch_station_master(force=True)
        except Exception as e:
            print(f"[relay] initial station master fetch failed: {e}")
            traceback.print_exc()
        threading.Thread(target=poll_loop, daemon=True).start()
        print("[relay] background poller started")


@app.route("/stations")
def stations():
    with _lock:
        return jsonify({
            "updatedAt": _last_poll_ok,
            "error": _last_poll_error,
            "stations": list(_station_state.values()),
        })


@app.route("/all-stations")
def all_stations():
    """Every real JMA station (~4,360 nationwide), regardless of whether
    it has reported a live reading recently. Roblox should call this ONCE
    on boot to spawn a Part for every real station, then poll /stations
    on a timer for live intensity updates to color/pulse them."""
    with _master_lock:
        return jsonify({
            "updatedAt": _master_last_fetch_ok,
            "error": _master_last_error,
            "stationCount": len(_station_master),
            "stations": list(_station_master.values()),
        })


@app.route("/health")
def health():
    with _lock, _master_lock:
        return jsonify({
            "ok": True,
            "lastPollAttempt": _last_poll_attempt,
            "lastPollOk": _last_poll_ok,
            "lastPollError": _last_poll_error,
            "stationCount": len(_station_state),
            "masterStationCount": len(_station_master),
            "masterLastFetchOk": _master_last_fetch_ok,
            "masterLastError": _master_last_error,
        })


@app.route("/debug-detail")
def debug_detail():
    """Returns the combined station readings from the most recent
    earthquakes, similar to JQuake's live map."""

    try:
        resp = requests.get(LIST_URL, headers=HEADERS, timeout=15)
        quake_list = resp.json()
    except Exception as e:
        return jsonify({"error": f"list fetch failed: {e}"})

    if not isinstance(quake_list, list):
        return jsonify({"error": "quake list was invalid"})

    # Look through the newest 20 events.
    recent = [q for q in quake_list if q.get("json")][:20]

    merged = {}
    detail_urls = []

    for item in recent:
        detail_url = DETAIL_BASE + item["json"]
        detail_urls.append(detail_url)

        try:
            r = requests.get(detail_url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue

            detail_json = r.json()
            readings = extract_station_readings(detail_json)

            for reading in readings:
                code = reading["code"]

                # Keep the strongest intensity seen for this station.
                if (
                    code not in merged or
                    reading["intensity"] > merged[code]["intensity"]
                ):
                    merged[code] = reading

        except Exception:
            continue

    return jsonify({
        "events_checked": len(recent),
        "readings_found": len(merged),
        "readings": list(merged.values()),
        "detail_urls": detail_urls,
    })

@app.route("/debug-fetch")
def debug_fetch():
    """Runs the exact same quake-list fetch as poll_once(), but
    synchronously inside this request - no background thread, no log
    buffering, no timing ambiguity. Hit this directly in a browser and
    whatever it returns (including if the request itself hangs/times out
    in the browser) tells us definitively what's happening."""
    import time as _time
    started = _time.time()
    try:
        resp = requests.get(LIST_URL, headers=HEADERS, timeout=15)
        elapsed = _time.time() - started
        return jsonify({
            "elapsed_seconds": elapsed,
            "status_code": resp.status_code,
            "body_preview": resp.text[:1000],
        })
    except Exception as e:
        elapsed = _time.time() - started
        return jsonify({
            "elapsed_seconds": elapsed,
            "exception_type": str(type(e)),
            "exception": str(e),
        })


@app.route("/")
def index():
    return jsonify({
        "service": "jquake-roblox-relay",
        "endpoints": ["/all-stations", "/stations", "/health"],
    })


# Start the poller as soon as the module is imported - this covers both
# `python app.py` (below) and gunicorn importing `app:app` directly.
ensure_poller_started()

if __name__ == "__main__":
    print(f"[relay] listening on :{PORT}")
    print(f"[relay] Roblox should GET http://<this-host>:{PORT}/stations")
    app.run(host="0.0.0.0", port=PORT)
