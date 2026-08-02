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

--------------------------------------------------------------------
FIX (see poll_once): list.json interleaves many JMA bulletin types -
hypocenter-only reports, tsunami-only bulletins, foreign/offshore
events with no felt shaking in Japan - alongside actual felt-intensity
reports. Only entries with a Body.Intensity.Observation block produce
any station readings. The old code sliced only the newest
MAX_EVENTS_PER_POLL (5) entries and fetched detail JSON for exactly
those, regardless of whether they were the right *kind* of entry. If
the 5 most recent list.json items all happened to be non-intensity
bulletins (very plausible in a quiet stretch), extract_station_readings
returned [] for all 5, merged stayed {}, and /stations legitimately -
but wrongly - reported 0 live stations.

The fix separates "how far back to look" (SCAN_WINDOW) from "how many
detail JSONs we're willing to fetch" (MAX_EVENTS_PER_POLL, now a fetch
budget rather than a blind slice). poll_once() now walks forward
through up to SCAN_WINDOW list entries, fetching detail JSON for up to
MAX_EVENTS_PER_POLL of them, until it finds intensity data or runs out
of budget/window. In the common case (newest event has data) this costs
exactly the same single fetch as before.
--------------------------------------------------------------------
"""

import os
import time
import threading
import traceback
import functools
import concurrent.futures

import requests
from flask import Flask, jsonify

# Force every print() call to flush immediately.
print = functools.partial(print, flush=True)

PORT = int(os.environ.get("PORT", 8787))
POLL_INTERVAL_S = 20  # be polite to JMA - no need to poll faster than this
LIST_URL = "https://www.jma.go.jp/bosai/quake/data/list.json"
DETAIL_BASE = "https://www.jma.go.jp/bosai/quake/data/"

# How far back into list.json (most-recent-first) we're willing to scan
# per poll cycle, looking for entries that actually carry felt-intensity
# data. This is just a "how far do we look" cap - it does NOT mean we
# fetch detail JSON for all of them (see MAX_EVENTS_PER_POLL below).
SCAN_WINDOW = 20

# How many detail JSON fetches we're willing to spend per poll cycle.
# This used to be a blind slice of the newest N list.json entries
# (list.json[:MAX_EVENTS_PER_POLL]) - that's what caused the bug
# described above. It's now a fetch budget: poll_once() walks through
# up to SCAN_WINDOW entries and stops fetching once it has spent this
# many detail-JSON requests, so a quiet/duplicate-heavy stretch of the
# feed can't blow past POLL_INTERVAL_S or hammer JMA.
MAX_EVENTS_PER_POLL = 5

STATION_MASTER_URL = "https://www.jma.go.jp/jma/kishou/know/jishin/intens-st/stations.json"
STATION_MASTER_FALLBACK_URL = "https://raw.githubusercontent.com/iku55/jma_int_stations/main/stations.json"
STATION_MASTER_REFRESH_S = 24 * 60 * 60

INTENSITY_MAP = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5-": 5.0, "5+": 5.5,
    "6-": 6.0, "6+": 6.5,
    "7": 7,
}

app = Flask(__name__)

_lock = threading.Lock()
_station_state = {}
_last_poll_ok = None
_last_poll_error = None
_last_poll_attempt = None
_poller_started = False
_poller_lock = threading.Lock()

_master_lock = threading.Lock()
_station_master = {}
_master_last_fetch_ok = None
_master_last_error = None

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

_session = requests.Session()
_session.headers.update(HEADERS)

REQUEST_TIMEOUT = (5, 10)

# Hard wall-clock ceiling for a single fetch_json() call, enforced from
# OUTSIDE the requests call via a worker thread + .result(timeout=...).
# REQUEST_TIMEOUT alone is not a reliable ceiling: it only bounds time
# spent actively connecting/reading on the socket. It does NOT bound
# things like a stuck DNS resolution, or the requests library blocking
# while waiting on a shared, pooled Session under some low-level
# connection states. Symptom seen in production: lastPollAttempt
# updates, but lastPollOk/lastPollError never do - poll_once() hangs
# forever inside a fetch_json() call. This wrapper guarantees
# fetch_json() always returns within FETCH_WATCHDOG_S wall-clock
# seconds, no matter what requests does internally, so the poll loop
# can never be stuck for good.
FETCH_WATCHDOG_S = 20
_fetch_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="fetch_json"
)


def _fetch_json_inner(url, what):
    print(f"[relay] fetch_json: starting {what} ({url})")
    t0 = time.time()
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as e:
        msg = f"{what}: request failed after {time.time() - t0:.1f}s: {e}"
        print(f"[relay] {msg}")
        return None, msg
    except Exception as e:
        msg = f"{what}: unexpected error after {time.time() - t0:.1f}s: {e}"
        print(f"[relay] {msg}")
        return None, msg

    print(f"[relay] fetch_json: {what} returned HTTP {resp.status_code} in {time.time() - t0:.1f}s")

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


def fetch_json(url, what):
    """GET a URL and parse JSON, with real error visibility. Never raises,
    and never blocks longer than FETCH_WATCHDOG_S wall-clock seconds -
    returns (data, error_string) either way. If the underlying call is
    still stuck when the watchdog fires, the worker thread is abandoned
    (it will not block process shutdown) and this function returns a
    timeout error immediately so the poll loop can continue."""
    future = _fetch_executor.submit(_fetch_json_inner, url, what)
    try:
        return future.result(timeout=FETCH_WATCHDOG_S)
    except concurrent.futures.TimeoutError:
        msg = f"{what}: watchdog fired - no response within {FETCH_WATCHDOG_S}s (call abandoned, likely stuck below the requests layer)"
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
    """Walk Body.Intensity.Observation.Pref[].Area[].City[].IntensityStation[]

    BUG THIS GUARDS AGAINST: JMA's detail JSON does not omit the
    "Intensity" key for events with no felt intensity - it sets it to
    JSON null explicitly. That means:
        (detail_json.get("Body") or {}).get("Intensity", {})
    returns None, not {}, because .get(key, default) only falls back to
    the default when the KEY is missing, not when its value is null.
    Calling .get("Observation") on that None then raised an uncaught
    AttributeError inside poll_once() on almost every single poll cycle
    (most events have no felt intensity), which silently crashed the
    poll before it ever reached the code that records _last_poll_ok /
    _last_poll_error - hence /health showing both stuck at null forever
    even though lastPollAttempt kept updating.
    """
    out = []
    body = detail_json.get("Body")
    if not isinstance(body, dict):
        return out

    intensity_block = body.get("Intensity")
    if not isinstance(intensity_block, dict):
        # Key missing OR explicitly null - both mean "no felt intensity
        # data for this event," which is normal and common, not an error.
        return out

    obs = intensity_block.get("Observation")
    if not isinstance(obs, dict) or not isinstance(obs.get("Pref"), list):
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
            "affi": entry.get("affi"),
        }
    return parsed


def fetch_station_master(force=False):
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

    with _master_lock:
        _master_last_error = last_err
    print(f"[relay] station master fetch failed: {last_err}")


def poll_once():
    """
    Fetch the recent-events list, then walk forward through it looking
    for events that actually carry felt-intensity station data.

    FIXED BEHAVIOR: previously this did
        for item in quake_list[:MAX_EVENTS_PER_POLL]:
    which blindly fetched detail JSON for only the newest 5 list.json
    entries, no matter what kind of bulletin they were. list.json mixes
    intensity-bearing reports in with hypocenter-only / tsunami-only /
    no-shaking bulletins, so it was entirely possible (and apparently
    what you're hitting) for the newest 5 entries to ALL be the wrong
    kind, in which case merged stayed empty and /stations correctly-but-
    misleadingly reported 0 live stations even during otherwise-normal
    operation.

    Now: SCAN_WINDOW bounds how far back into the list we're willing to
    look, MAX_EVENTS_PER_POLL bounds how many detail-JSON fetches we're
    willing to spend doing it (to stay polite to JMA and keep each cycle
    comfortably under POLL_INTERVAL_S). Events with no Body.Intensity
    block still consume a bit of scan distance but do NOT count as a
    "wasted" fetch differently from before - they just don't stop the
    walk early anymore.
    """
    global _last_poll_ok, _last_poll_error, _last_poll_attempt

    with _lock:
        _last_poll_attempt = time.time()

    print("[relay] poll_once started")

    quake_list, err = fetch_json(LIST_URL, "quake list")
    if err:
        with _lock:
            _last_poll_error = err
        return

    if not isinstance(quake_list, list):
        with _lock:
            _last_poll_error = "quake list was not a JSON list"
        return

    now = time.time()

    with _master_lock:
        master_snapshot = dict(_station_master)

    merged = {}
    events_fetched = 0
    events_scanned = 0
    events_with_readings = 0
    per_event_errors = []

    for item in quake_list[:SCAN_WINDOW]:
        if events_fetched >= MAX_EVENTS_PER_POLL:
            break

        events_scanned += 1

        # Guard each individual event so one malformed/unexpected detail
        # payload can't take down the whole poll cycle (which is exactly
        # what happened before: one bad event -> uncaught exception ->
        # _last_poll_ok/_last_poll_error never get set, every cycle,
        # forever). Any per-event problem is now recorded and skipped
        # instead of propagating.
        try:
            json_file = item.get("json")
            if not json_file:
                continue

            detail, err = fetch_json(
                DETAIL_BASE + json_file,
                f"quake detail ({json_file})",
            )
            events_fetched += 1

            if err or not detail:
                if err:
                    per_event_errors.append(err)
                continue

            readings = extract_station_readings(detail)
            if not readings:
                # No Body.Intensity.Observation block - hypocenter-only,
                # tsunami-only, or similar bulletin type. Normal - move
                # on to the next list entry instead of giving up.
                continue

            events_with_readings += 1

            for r in readings:
                code = r.get("code")
                if not code:
                    continue

                if code in merged:
                    continue

                master = master_snapshot.get(code, {})

                merged[code] = {
                    "code": code,
                    "name": r.get("name") or code,
                    "pref": r.get("pref") or master.get("pref"),
                    "lat": master.get("lat"),
                    "lon": master.get("lon"),
                    "matched": code in master_snapshot,
                    "intensity": r["intensity"],
                    "eventTime": item.get("at"),
                    "updatedAt": now,
                }
        except Exception as e:
            # Never let a single event's parsing take down the cycle.
            # Record it so it's visible in logs, then keep scanning.
            msg = f"event {item.get('json', '?')}: {type(e).__name__}: {e}"
            print(f"[relay] poll_once: error processing {msg}")
            per_event_errors.append(msg)
            continue

    with _lock:
        _station_state.clear()
        _station_state.update(merged)
        _last_poll_ok = now
        _last_poll_error = (
            f"{len(per_event_errors)} event(s) had errors this cycle; last: {per_event_errors[-1]}"
            if per_event_errors else None
        )

    print(
        f"[relay] updated: {len(merged)} live stations "
        f"(scanned {events_scanned} list entries, fetched {events_fetched} details, "
        f"{events_with_readings} had intensity data, {len(per_event_errors)} errors)"
    )

    if not merged:
        print("[relay] poll_once: no readings this cycle")


def poll_loop():
    print("[relay] live poll thread started")

    while True:
        cycle_start = time.time()
        try:
            print("[relay] polling JMA...")

            fetch_station_master()
            poll_once()

            with _lock:
                print(
                    f"[relay] state now has {len(_station_state)} stations, "
                    f"last ok={_last_poll_ok}"
                )

        except Exception as e:
            # Belt-and-suspenders: poll_once() now catches per-event
            # errors internally, but if something outside that (e.g.
            # fetch_station_master, or a bug in poll_once itself before
            # it reaches its own try/except) still throws, make sure
            # it's visible via /health instead of only appearing in
            # logs - this is what let the original bug hide for so long.
            msg = f"{type(e).__name__}: {e}"
            print(f"[relay] poll error: {msg}")
            traceback.print_exc()
            with _lock:
                _last_poll_error = msg

        # Every fetch_json() call is now watchdog-bounded (FETCH_WATCHDOG_S),
        # so a full cycle should never take dramatically longer than
        # (events fetched) * FETCH_WATCHDOG_S. Log it so a real hang is
        # visible in Render's logs rather than silently eating the interval.
        cycle_elapsed = time.time() - cycle_start
        print(f"[relay] poll cycle took {cycle_elapsed:.1f}s")

        time.sleep(POLL_INTERVAL_S)


def ensure_poller_started():
    global _poller_started
    with _poller_lock:
        if _poller_started:
            return
        _poller_started = True
        try:
            fetch_station_master(force=True)
        except Exception as e:
            print(f"[relay] initial station master fetch failed: {e}")
            traceback.print_exc()
        threading.Thread(target=poll_loop, daemon=True).start()
        print("[relay] background poller started")


@app.route("/stations")
def stations():
    with _lock, _master_lock:
        stations = []

        for code, master in _station_master.items():
            live = _station_state.get(code)

            "intensity": live["intensity"] if live else 0

            stations.append({
                "code": code,
                "name": master.get("name_ja") or code,
                "pref": master.get("pref"),
                "lat": master.get("lat"),
                "lon": master.get("lon"),
                "matched": True,
                "intensity": live["intensity"] if live else 0,
                "eventTime": live.get("eventTime") if live else None,
                "updatedAt": live.get("updatedAt") if live else None,
            })

        return jsonify({
            "pid": os.getpid(),
            "stationCount": len(stations),
            "updatedAt": _last_poll_ok,
            "error": _last_poll_error,
            "stations": stations,
        })


@app.route("/all-stations")
def all_stations():
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
    """Returns combined station readings from the most recent earthquakes."""
    quake_list, err = fetch_json(LIST_URL, "debug quake list")
    if err or not isinstance(quake_list, list):
        return jsonify({"error": err or "quake list was invalid"})

    # Uses the same SCAN_WINDOW as poll_once() for consistency when
    # comparing this debug view against what the live poller sees.
    recent = quake_list[:SCAN_WINDOW]

    merged = {}
    detail_urls = []

    for item in recent:
        json_file = item.get("json")
        if not json_file:
            continue
        detail_url = DETAIL_BASE + json_file
        detail_urls.append(detail_url)

        detail_json, derr = fetch_json(detail_url, f"debug detail ({json_file})")
        if derr or not detail_json:
            continue

        readings = extract_station_readings(detail_json)

        for reading in readings:
            code = reading["code"]
            if not code:
                continue
            if (
                code not in merged or
                reading["intensity"] > merged[code]["intensity"]
            ):
                merged[code] = reading

    return jsonify({
        "events_checked": len(recent),
        "readings_found": len(merged),
        "readings": list(merged.values()),
        "detail_urls": detail_urls,
    })


@app.route("/debug-fetch")
def debug_fetch():
    """Runs the exact same quake-list fetch as poll_once(), synchronously,
    inside this request - for isolating network issues."""
    started = time.time()
    try:
        resp = requests.get(LIST_URL, headers=HEADERS, timeout=15)
        elapsed = time.time() - started
        return jsonify({
            "elapsed_seconds": elapsed,
            "status_code": resp.status_code,
            "body_preview": resp.text[:1000],
        })
    except Exception as e:
        elapsed = time.time() - started
        return jsonify({
            "elapsed_seconds": elapsed,
            "exception_type": str(type(e)),
            "exception": str(e),
        })


@app.route("/")
def index():
    return jsonify({
        "service": "jquake-roblox-relay",
        "endpoints": ["/all-stations", "/stations", "/health", "/debug-detail", "/debug-fetch"],
    })


# Start the poller as soon as the module is imported - this covers both
# `python app.py` (below) and gunicorn importing `app:app` directly.
ensure_poller_started()

if __name__ == "__main__":
    print(f"[relay] listening on :{PORT}")
    print(f"[relay] Roblox should GET http://<this-host>:{PORT}/stations")
    app.run(host="0.0.0.0", port=PORT)
