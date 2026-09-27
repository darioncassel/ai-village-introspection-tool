"""Village days: time ranges from village-transcript.json, the day lookup, its cache, and failing loudly
when the days can't be built. Synthetic transcripts only, except test_real_days_match_transcript, which
runs only with VILLAGE_DATA_TESTS=1 (it parses the real transcript and events, ~3GB, ~1 min)."""

import gzip
import http.client
import json
import os
import threading
import time

import pytest

from village_introspect import cc_lib as cc
from village_introspect import config, server
from village_introspect import tierb_lib as tb
from village_introspect import village_lib as vl


def _ev(ts, typ="AGENT_TALK"):
    return {"timestamp": ts, "time": ts[11:19], "type": typ}


# The shapes found in the real transcript: an evening session that runs past 00:00 UTC (day 78), a
# Friday whose last events fall on a Saturday UTC date (day 83), an empty day row (54), and a one-event
# stub day sharing its date with the real day (439 / 440, also the last day here).
TRANSCRIPT = {
    "village": {"id": "v", "name": "test"},
    "days": [
        {"day": 54, "date": "2025-06-17", "events": []},
        {"day": 78, "date": "2025-06-18", "events": [_ev("2025-06-18T17:59:34.130Z"), _ev("2025-06-19T04:01:00.000Z")]},
        {"day": 79, "date": "2025-06-19", "events": [_ev("2025-06-19T18:00:00.000Z"), _ev("2025-06-19T22:00:00.000Z")]},
        {"day": 83, "date": "2025-06-20", "events": [_ev("2025-06-20T18:00:00.000Z"), _ev("2025-06-21T00:03:33.100Z")]},
        {"day": 439, "date": "2026-06-15", "events": [_ev("2026-06-15T13:03:30.330Z")]},
        {
            "day": 440,
            "date": "2026-06-15",
            "events": [_ev("2026-06-15T16:00:00.861Z"), _ev("2026-06-15T21:05:01.390Z")],
        },
    ],
}


@pytest.fixture()
def ds(tmp_path, monkeypatch):
    """A dataset dir holding TRANSCRIPT, a fresh state dir, and day ranges loaded from them on first use."""
    d = tmp_path / "ds"
    d.mkdir()
    (d / "village-transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    monkeypatch.setattr(config, "DATASET", d)
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    cc.set_day_ranges(None)
    return d


def _stamp(d):
    st = (d / "village-transcript.json").stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


# ---- timestamps -----------------------------------------------------------------------------------
def test_utc_seconds_formats():
    base = config.utc_seconds("2026-06-15 16:00:00")
    assert base == config.utc_seconds("2026-06-15T16:00:00Z") == config.utc_seconds("2026-06-15T18:00:00+02:00")
    assert config.utc_seconds("2026-06-15 16:00:00.5") == base + 0.5  # any number of fraction digits
    assert config.utc_seconds("2026-06-15 16:00:00.57899") == pytest.approx(base + 0.57899)
    assert config.utc_seconds("2026-06-15T16:00:00.861Z") == pytest.approx(base + 0.861)
    for bad in (None, "", "2026-06-15", "garbage", 17, "2026-02-30 10:00:00"):
        assert config.utc_seconds(bad) is None


# ---- day assignment -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "created_at,day",
    [
        ("2025-06-18 17:59:34.130512", 78),  # the day's first event (the transcript cuts it to ms)
        ("2025-06-19 01:00:00.123456", 78),  # evening session after 00:00 UTC (was: day 79)
        ("2025-06-19 04:01:00.000999", 78),  # its last event
        ("2025-06-19 09:00:00", 78),  # overnight, between two days -> the day before
        ("2025-06-19 18:00:00.5", 79),
        ("2025-06-21 00:00:02.5", 83),  # Friday's pause on a Saturday UTC date (was: None)
        ("2025-06-22 12:00:00", 83),  # weekend -> the day before
        ("2026-06-15 13:03:30.330999", 439),  # the stub day
        ("2026-06-15 15:00:00", 439),  # between the stub and the real day
        ("2026-06-15 16:30:00", 440),  # the real day on the shared date (was: 439)
        ("2026-06-15 21:05:01.390400", 440),  # the last day's last event
        ("2026-06-15 21:05:03", None),  # after the last day
        ("2025-06-18 17:00:00", None),  # before the first day
        ("", None),
        (None, None),
        ("not a time", None),
    ],
)
def test_village_day_by_time_not_date(ds, created_at, day):
    assert cc.village_day(created_at) == day


def test_day_ranges_shape_and_dates(ds):
    rs = config.day_ranges(ds)
    assert [r["day"] for r in rs] == [78, 79, 83, 439, 440]  # the empty day 54 has no range
    assert cc.day_dates() == {
        78: "2025-06-18",
        79: "2025-06-19",
        83: "2025-06-20",
        439: "2026-06-15",
        440: "2026-06-15",
    }
    assert rs[0]["start"] == config.utc_seconds("2025-06-18T17:59:34.130Z")
    assert rs[0]["end"] == config.utc_seconds("2025-06-19T04:01:00.000Z")


def test_malformed_day_rows_are_skipped_with_a_warning(ds, capsys):
    t = json.loads(json.dumps(TRANSCRIPT))
    t["days"] += [
        "not a row",
        {"date": "2026-06-16", "events": [_ev("2026-06-16T16:00:00.000Z")]},  # no day number
        {"day": 441, "date": "2026-06-16", "events": [{"timestamp": "never"}]},  # no usable timestamp
        {"day": 442, "events": [_ev("2026-06-17T16:00:00.000Z")]},  # no date: taken from its first event
    ]
    (ds / "village-transcript.json").write_text(json.dumps(t), encoding="utf-8")
    rs = config.day_ranges(ds)
    assert [r["day"] for r in rs] == [78, 79, 83, 439, 440, 442] and rs[-1]["date"] == "2026-06-17"
    assert "skipped 3 malformed day rows" in capsys.readouterr().err


def test_cc_window_day(ds):
    """A window goes to the day of its first row; one starting between two days goes to the nearer."""

    def win(seq, ts):
        return [
            cc.CCRow(seq=seq, kind="init", role="system", text="{}", created_at=ts),
            cc.CCRow(seq=seq + 1, kind="text", role="assistant", text="hi", created_at=ts),
        ]

    rows = (
        win(0, "2025-06-19 04:05:00")  # 4 min after day 78 ended
        + win(2, "2025-06-19 16:00:00")  # 2h before day 79 starts, 12h after day 78 ended
        + win(4, "2025-06-19 19:00:00")  # inside day 79
        + win(6, "2026-06-15 22:00:00")  # after the last day
    )
    assert [(s["seg_id"], s["day"]) for s in cc.segments(rows)] == [(0, 78), (1, 79), (2, 79), (3, None)]


# ---- cache ---------------------------------------------------------------------------------------
def test_cache_uses_a_new_file_and_ignores_the_old_date_map(ds):
    cache = config.STATE_DIR / "cache"
    cache.mkdir(parents=True)
    old = cache / "date_day_map.json"  # the earlier date -> day map, same stamp: must not be read
    old.write_text(json.dumps({"stamp": {"size": 1, "mtime": 1}, "map": {"2025-06-18": 1}}), encoding="utf-8")
    assert [r["day"] for r in config.day_ranges(ds)] == [78, 79, 83, 439, 440]
    c = json.loads((cache / "day_ranges.json").read_text(encoding="utf-8"))
    assert c["stamp"] == _stamp(ds) and [r["day"] for r in c["days"]] == [78, 79, 83, 439, 440]
    assert old.read_text(encoding="utf-8").startswith('{"stamp"')  # left alone
    assert not list(cache.glob("*.tmp"))


@pytest.mark.parametrize(
    "bad",
    [
        b"not json",
        b"\xff\xfe\x00",  # not UTF-8
        b"[]",
        b"{}",
        json.dumps({"version": 99, "stamp": "STAMP", "days": []}).encode(),
        json.dumps({"version": 1, "stamp": "STAMP", "days": {"78": 1}}).encode(),
        json.dumps({"version": 1, "stamp": "STAMP", "days": []}).encode(),
        json.dumps(
            {"version": 1, "stamp": "STAMP", "days": [{"day": "78", "date": "d", "start": 1, "end": 2}]}
        ).encode(),
        json.dumps(
            {"version": 1, "stamp": "STAMP", "days": [{"day": True, "date": "d", "start": 1, "end": 2}]}
        ).encode(),
        json.dumps({"version": 1, "stamp": "STAMP", "days": [{"day": 1, "date": "d", "start": 3, "end": 2}]}).encode(),
        json.dumps(
            {"version": 1, "stamp": "STAMP", "days": [{"day": 1, "date": "d", "start": "1", "end": 2}]}
        ).encode(),
        json.dumps(
            {
                "version": 1,
                "stamp": "STAMP",
                "days": [{"day": 2, "date": "d", "start": 5, "end": 6}, {"day": 1, "date": "d", "start": 1, "end": 2}],
            }
        ).encode(),
        json.dumps({"version": 1, "stamp": "STAMP", "days": ["x"]}).encode(),
    ],
)
def test_ill_shaped_cache_is_a_miss(ds, bad):
    cache = config.STATE_DIR / "cache" / "day_ranges.json"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(bad.replace(b'"STAMP"', json.dumps(_stamp(ds)).encode()))
    assert [r["day"] for r in config.day_ranges(ds)] == [78, 79, 83, 439, 440]  # rebuilt, not crashed
    assert [r["day"] for r in json.loads(cache.read_text(encoding="utf-8"))["days"]] == [78, 79, 83, 439, 440]


def test_cache_write_failure_keeps_the_ranges(ds, capsys):
    config.STATE_DIR.mkdir()
    (config.STATE_DIR / "cache").write_text("a file where the cache dir should be")
    assert [r["day"] for r in cc.load_days()] == [78, 79, 83, 439, 440]
    assert "could not cache the day index" in capsys.readouterr().err
    assert cc.village_day("2026-06-15 16:30:00") == 440


# ---- failing loudly ------------------------------------------------------------------------------
def test_missing_transcript_raises(ds):
    (ds / "village-transcript.json").unlink()
    with pytest.raises(RuntimeError, match="can't read"):
        cc.load_days()


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(TRANSCRIPT)[:200],  # a truncated download
        json.dumps({"days": {}}),
        json.dumps([1, 2]),
        json.dumps({"days": [{"day": 1, "date": "2025-04-02", "events": []}]}),  # no event anywhere
    ],
)
def test_unusable_transcript_raises_and_the_failure_is_not_kept(ds, text):
    good = (ds / "village-transcript.json").read_text(encoding="utf-8")
    (ds / "village-transcript.json").write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="village-transcript.json"):
        cc.load_days()
    with pytest.raises(RuntimeError):
        cc.village_day("2026-06-15 16:30:00")  # loud, not a silent None
    (ds / "village-transcript.json").write_text(good, encoding="utf-8")
    assert cc.village_day("2026-06-15 16:30:00") == 440  # retried once the file is fixed


# ---- warm() --------------------------------------------------------------------------------------
@pytest.fixture()
def fresh_vl(monkeypatch):
    """village_lib with no index and a fresh STATUS; the corpora come from preloaded caches."""
    monkeypatch.setattr(vl, "STATUS", {"stage": "starting", "t0": time.time(), "error": None})
    monkeypatch.setattr(tb, "_ROSTER", {"a1": {"agent_id": "a1", "name": "Claude Opus 4.8", "family": "Anthropic"}})
    monkeypatch.setattr(tb, "_ROOM_NAMES", {"r": "general"})
    monkeypatch.setattr(tb, "_ACTIVITY", [])
    monkeypatch.setattr(cc, "_ALL_ROWS", [])
    vl.set_index(None)
    yield
    vl.set_index(None)


def _chat(ei, created_at, day):
    return {
        "ei": ei,
        "room": "r",
        "sid": "a1",
        "speaker": "Claude Opus 4.8",
        "family": "Anthropic",
        "is_agent": True,
        "content": "evening update",
        "day": day,
        "created_at": created_at,
        "thinking": "",
        "think_kind": "",
    }


def test_warm_fails_loudly_on_a_bad_transcript(ds, fresh_vl, monkeypatch):
    (ds / "village-transcript.json").write_text(json.dumps(TRANSCRIPT)[:300], encoding="utf-8")
    monkeypatch.setattr(tb, "_CHAT", [_chat(1, "2025-06-19 01:00:00", None)])
    with pytest.raises(RuntimeError):
        vl.warm()
    st = vl.status()
    assert st["stage"] == "error" and not st["ready"]
    assert st["error"].startswith("RuntimeError: can't build the village days") and "download it again" in st["error"]


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, json.loads(r.read())
    finally:
        conn.close()


def test_warm_fails_when_chat_has_no_days(ds, fresh_vl, monkeypatch):
    monkeypatch.setattr(tb, "_CHAT", [_chat(1, "2027-01-01 12:00:00", None)])  # outside every day
    with pytest.raises(RuntimeError, match="0 days"):
        vl.warm()
    st = vl.status()
    assert st["stage"] == "error" and "0 days" in st["error"]
    assert not st["ready"] and not vl.ready()  # the day-less index is never installed
    # so the server keeps refusing timeline routes and the viewer shows the error, not an empty timeline
    srv = server.Server(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        code, body = _get(port, "/api/status")
        assert code == 200 and body["stage"] == "error" and not body["ready"]
        code, body = _get(port, "/api/overview")
        assert code == 503 and "0 days" in body["detail"]
    finally:
        srv.shutdown()
        srv.server_close()
    with pytest.raises(RuntimeError, match="0 days"):  # a lazy caller gets the same error
        vl.overview()
    assert not vl.ready()


def test_warm_ready_puts_evening_posts_on_their_day(ds, fresh_vl, monkeypatch, capsys):
    ca = "2025-06-19 01:00:00.123456"
    late = "2026-06-16 12:00:00"  # after the last day: warned about, not on the timeline
    monkeypatch.setattr(tb, "_CHAT", [_chat(1, ca, cc.village_day(ca)), _chat(2, late, cc.village_day(late))])
    vl.warm()
    assert vl.status()["stage"] == "ready"
    ov = vl.overview()
    assert [(d["day"], d["date"]) for d in ov["days"]] == [(78, "2025-06-18")]  # the transcript's date, not 06-19
    assert "1 chat messages and 0 activity events fall outside every village day" in capsys.readouterr().err


# ---- goal eras on a shared date ------------------------------------------------------------------
def test_era_announcement_found_on_the_real_day_sharing_the_start_date(monkeypatch):
    monkeypatch.setattr(vl, "GOAL_ERAS", [("2026-06-08", "Event org")])
    chat = [
        dict(_chat(1, "2026-06-08 14:30:03", 432), is_agent=False, sid=None, speaker="automated", content="resume"),
        dict(_chat(2, "2026-06-08 16:00:00", 433), is_agent=False, sid=None, speaker="Shoshannah", content="New goal!"),
    ]
    ros = {"a1": {"agent_id": "a1", "name": "Claude Opus 4.8", "family": "Anthropic"}}
    ix = vl._build_from(chat, [], {}, ros, [], [], {432: "2026-06-08", 433: "2026-06-08"})
    (era,) = ix["overview"]["eras"]
    assert (era["start_day"], era["end_day"], era["announce_ei"]) == (432, 433, 2)


def test_date_jump_goes_to_the_busiest_day_on_a_shared_date(fresh_vl):
    chat = [
        dict(_chat(1, "2026-06-15 13:03:30", 439), is_agent=False, sid=None, speaker="automated", content="resume"),
        _chat(2, "2026-06-15 16:00:01", 440),
        _chat(3, "2026-06-15 16:05:00", 440),
    ]
    ros = {"a1": {"agent_id": "a1", "name": "Claude Opus 4.8", "family": "Anthropic"}}
    vl.set_index(vl._build_from(chat, [], {}, ros, [], [], {439: "2026-06-15", 440: "2026-06-15"}))
    assert [d["day"] for d in vl.overview()["days"]] == [439, 440]
    assert vl.search("2026-06-15")["jump"] == {"day": 440}  # not the one-event stub that comes first
    with pytest.raises(ValueError, match="no village activity"):
        vl.search("2026-06-16")


# ---- real data (opt-in) ----------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("VILLAGE_DATA_TESTS") != "1", reason="set VILLAGE_DATA_TESTS=1 (parses the real transcript)"
)
def test_real_days_match_transcript(tmp_path, monkeypatch):
    """Every event that appears in the transcript gets the transcript's own day, and no chat message
    falls outside every day."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)  # never write into the user's state dir
    cc.set_day_ranges(None)
    t = json.loads((config.DATASET / "village-transcript.json").read_text(encoding="utf-8"))
    tx = {}
    for d in t["days"]:
        for e in d["events"]:
            tx.setdefault((e["timestamp"][:23].replace("T", " "), e["type"]), set()).add(d["day"])
    del t
    days = {r["day"] for r in cc.load_days()}
    assert {55, 89, 433, 440, 454} <= days
    matched = wrong = no_day_chat = 0
    with gzip.open(config.DATASET / "events.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            at = (r.get("data") or {}).get("actionType")
            if not at:
                continue
            ca = r.get("created_at") or ""
            day = cc.village_day(ca)
            no_day_chat += day is None and at in ("AGENT_TALK", "USER_TALK")
            want = tx.get((ca[:23], at))
            if want:
                matched += 1
                wrong += day not in want
    print(f"\n{matched} events matched to the transcript, {wrong} on a different day, {no_day_chat} chat without a day")
    assert matched > 300_000 and wrong == 0 and no_day_chat == 0
