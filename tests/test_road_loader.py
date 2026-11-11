"""Tests for the ROAD loader using synthetic candump lines and metadata fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canids.ingestion.road_loader import (
    DEFAULT_METADATA_KEYS,
    InjectionSpec,
    classify_capture,
    label_attack_frames,
    load_road_dir,
    parse_candump_file,
    parse_candump_line,
    parse_injection_spec,
    payload_matches,
)
from canids.ingestion.schema import ParseError, validate_frame_df

KEYS = DEFAULT_METADATA_KEYS


# ------------------------------------------------------------------ candump lines
def test_candump_standard_frame() -> None:
    ts, cid, dlc, payload = parse_candump_line("(1592592355.150553) can0 0C1#00112233445566FF")
    assert ts == pytest.approx(1592592355.150553)
    assert (cid, dlc) == (0x0C1, 8)
    assert payload == (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0xFF)


def test_candump_extended_and_short() -> None:
    _, cid, dlc, payload = parse_candump_line("(1.0) vcan0 18FEF100#AABB")
    assert cid == 0x18FEF100 and dlc == 2 and payload[2] == -1


def test_candump_empty_payload() -> None:
    _, _, dlc, payload = parse_candump_line("(1.0) can0 123#")
    assert dlc == 0 and payload == (-1,) * 8


@pytest.mark.parametrize(
    "line, reason",
    [
        ("", "empty_line"),
        ("not a candump line", "bad_format"),
        ("(abc) can0 123#00", "bad_timestamp"),
        ("(1.0) can0 123##1001122", "unsupported_frame"),
        ("(1.0) can0 123#R", "unsupported_frame"),
        ("(1.0) can0 123#001", "bad_hex"),
        ("(1.0) can0 123#00GG", "bad_hex"),
        ("(1.0) can0 123#000000000000000000", "bad_dlc"),
        ("(1.0) can0 FFFFFFFFF#00", "bad_id"),
    ],
)
def test_candump_malformed(line: str, reason: str) -> None:
    with pytest.raises(ParseError) as err:
        parse_candump_line(line)
    assert err.value.reason == reason


# ------------------------------------------------------------------ metadata
def test_parse_spec_targeted() -> None:
    spec = parse_injection_spec(
        {"injection_id": "0x0D0", "injection_data_str": "XXXXFFFF", "injection_interval": [1, 2]},
        KEYS,
    )
    assert spec.can_id == 0x0D0
    assert spec.data_pattern == "XXXXFFFF" + "X" * 8
    assert spec.intervals == ((1.0, 2.0),)
    assert spec.is_precise


def test_parse_spec_wildcard_id_and_multiple_intervals() -> None:
    spec = parse_injection_spec(
        {
            "injection_id": "XXX",
            "injection_data_str": "FFFFFFFFFFFFFFFF",
            "injection_interval": [[1, 2], [5, 6]],
        },
        KEYS,
    )
    assert spec.can_id is None and spec.is_precise and len(spec.intervals) == 2


@pytest.mark.parametrize(
    "entry, msg",
    [
        ({"injection_id": None, "injection_interval": [0, 10]}, "null"),
        ({"injection_id": "0D0"}, "no injection interval"),
        ({"injection_id": "0D0", "injection_interval": [5, 1]}, "before start"),
        (
            {"injection_id": "0D0", "injection_data_str": "ZZ", "injection_interval": [0, 1]},
            "pattern",
        ),
    ],
)
def test_parse_spec_rejects(entry: dict, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        parse_injection_spec(entry, KEYS)


def test_fully_wildcard_spec_is_not_precise() -> None:
    assert not InjectionSpec(((0.0, 1.0),), None, "X" * 16).is_precise


def test_classify_capture() -> None:
    fam = {"fuzzing": "Fuzzy", "max_speedometer": "Spoofing"}
    assert classify_capture("fuzzing_attack_1", fam) == ("Fuzzy", "fuzzing")
    assert classify_capture("max_speedometer_attack_2_masquerade", fam) == (
        "Spoofing",
        "max_speedometer_masquerade",
    )
    assert classify_capture("something_else", fam) is None


# ------------------------------------------------------------------ labelling
LOG = """(100.0) can0 0D0#0000000000000000
(100.5) can0 0D0#1111FFFF22223333
(101.5) can0 0D0#0000FFFF00000000
(101.6) can0 0D0#0000000000000000
(101.7) can0 1A0#0000FFFF00000000
(103.0) can0 0D0#0000FFFF00000000
"""


def test_label_attack_frames_id_data_and_window(tmp_path: Path) -> None:
    f = tmp_path / "x.log"
    f.write_text(LOG, encoding="utf-8")
    df, _ = parse_candump_file(f)
    spec = InjectionSpec(((0.4, 2.0),), 0x0D0, "XXXXFFFF" + "X" * 8)
    out = label_attack_frames(df, spec, "Spoofing", "max_speedometer")
    # row1: in window + ID + data -> attack; row2: same; row3: data mismatch; row4: wrong ID;
    # row5: outside window; row0: outside window
    assert out["label"].tolist() == ["Normal", "Spoofing", "Spoofing", "Normal", "Normal", "Normal"]
    validate_frame_df(out)


def test_payload_matches_nibbles(tmp_path: Path) -> None:
    f = tmp_path / "x.log"
    f.write_text("(1.0) can0 001#A5\n(1.1) can0 001#A6\n(1.2) can0 001#\n", encoding="utf-8")
    df, _ = parse_candump_file(f)
    assert payload_matches(df, "A" + "X" * 15).tolist() == [True, True, False]
    assert payload_matches(df, "X5" + "X" * 14).tolist() == [True, False, False]


# ------------------------------------------------------------------ directory
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_road_dir_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "road"
    _write(root / "ambient" / "ambient_dyno_drive_basic_long.log", LOG)
    _write(root / "attacks" / "max_speedometer_attack_1.log", LOG)
    _write(root / "attacks" / "fuzzing_attack_1.log", LOG + "(101.8) can0 7AB#FFFFFFFFFFFFFFFF\n")
    _write(root / "attacks" / "accelerator_attack_drive_1.log", LOG)
    _write(root / "attacks" / "mystery_attack_1.log", LOG)
    _write(root / "attacks" / "not_in_metadata_fuzzing.log", LOG)
    _write(root / "signal_extractions" / "ignored.log", LOG)
    meta = {
        "max_speedometer_attack_1": {
            "injection_id": "0x0D0",
            "injection_data_str": "XXXXFFFF",
            "injection_interval": [0.4, 2.0],
        },
        "fuzzing_attack_1": {
            "injection_id": "XXX",
            "injection_data_str": "FFFFFFFFFFFFFFFF",
            "injection_interval": [0.0, 5.0],
        },
        "accelerator_attack_drive_1": {"injection_id": None, "injection_interval": [0, 99]},
    }
    _write(root / "attacks" / "capture_metadata.json", json.dumps(meta))

    results = {r.source_file: (df, r) for df, r in load_road_dir(root)}
    assert "signal_extractions/ignored.log" not in results

    df, rep = results["ambient/ambient_dyno_drive_basic_long.log"]
    assert rep.kept_per_label == {"Normal": 6}

    df, rep = results["attacks/max_speedometer_attack_1.log"]
    assert rep.kept_per_label == {"Normal": 4, "Spoofing": 2}

    df, rep = results["attacks/fuzzing_attack_1.log"]
    assert rep.kept_per_label == {"Normal": 6, "Fuzzy": 1}

    for skipped in (
        "attacks/accelerator_attack_drive_1.log",
        "attacks/mystery_attack_1.log",
        "attacks/not_in_metadata_fuzzing.log",
    ):
        assert results[skipped][0] is None and results[skipped][1].skipped
