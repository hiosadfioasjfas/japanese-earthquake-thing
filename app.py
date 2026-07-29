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

import requests
from flask import Flask, jsonify

PORT = int(os.environ.get("PORT", 8787))
POLL_INTERVAL_S = 20  # be polite to JMA - no need to poll faster than this
LIST_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"
DETAIL_BASE = "https://www.jma.go.jp/bosai/quake/data/"
DECAY_HALF_LIFE_S = 5 * 60  # readings fade back toward 0 over ~5 minutes

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
_poller_started = False
_poller_lock = threading.Lock()

_master_lock = threading.Lock()
_station_master = {}       # code -> { code, name, name_ja, pref, lat, lon, affi }
_master_last_fetch_ok = None
_master_last_error = None

HEADERS = {"User-Agent": "roblox-jquake-relay/1.0 (personal project)"}


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
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            parsed = _parse_master_payload(resp.json())
            if not parsed:
                raise ValueError("parsed station master was empty")
            with _master_lock:
                _station_master.clear()
                _station_master.update(parsed)
                _master_last_fetch_ok = time.time()
                _master_last_error = None
            print(f"[relay] station master loaded: {len(parsed)} stations (source: {url})")
            return
        except Exception as e:
            last_err = f"{url}: {e}"
            continue

    # both sources failed
    with _master_lock:
        _master_last_error = last_err
    print(f"[relay] station master fetch failed: {last_err}")


def poll_once():
    global _last_poll_ok, _last_poll_error

    resp = requests.get(LIST_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    quake_list = resp.json()

    recent = [item for item in quake_list if item.get("json")][:15]
    now = time.time()

    for item in recent:
        detail_url = DETAIL_BASE + item["json"]
        try:
            d_resp = requests.get(detail_url, headers=HEADERS, timeout=10)
            d_resp.raise_for_status()
            detail = d_resp.json()
        except Exception:
            continue

        readings = extract_station_readings(detail)
        if not readings:
            continue

        with _master_lock:
            master_snapshot = dict(_station_master)

        with _lock:
            for r in readings:
                code = r.get("code")
                if not code:
                    continue
                existing = _station_state.get(code, {})
                master = master_snapshot.get(code)
                _station_state[code] = {
                    "code": code,
                    "name": r.get("name") or existing.get("name") or code,
                    "pref": r.get("pref") or existing.get("pref") or (master or {}).get("pref"),
                    "lat": (master or {}).get("lat", existing.get("lat")),
                    "lon": (master or {}).get("lon", existing.get("lon")),
                    "matched": master is not None,  # True if this code was found in the real JMA master
                    "intensity": r["intensity"],
                    "eventTime": item.get("at"),
                    "updatedAt": now,
                }
        break  # only need the most recent event that had readings

    with _lock:
        _last_poll_ok = now
        _last_poll_error = None


def decay_loop():
    while True:
        time.sleep(10)
        now = time.time()
        with _lock:
            for code, s in list(_station_state.items()):
                if s["intensity"] <= 0:
                    continue
                age = now - s["updatedAt"]
                if age <= 0:
                    continue
                decayed = s["intensity"] * math.pow(0.5, age / DECAY_HALF_LIFE_S)
                if decayed < 0.05:
                    s["intensity"] = 0
                else:
                    s["intensity"] = decayed


def poll_loop():
    global _last_poll_error
    while True:
        try:
            fetch_station_master()  # no-op if already fresh; retries if it failed before
            poll_once()
        except Exception as e:
            with _lock:
                _last_poll_error = str(e)
            print(f"[relay] poll failed: {e}")
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
        threading.Thread(target=poll_loop, daemon=True).start()
        threading.Thread(target=decay_loop, daemon=True).start()
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
            "lastPollOk": _last_poll_ok,
            "lastPollError": _last_poll_error,
            "stationCount": len(_station_state),
            "masterStationCount": len(_station_master),
            "masterLastFetchOk": _master_last_fetch_ok,
            "masterLastError": _master_last_error,
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
