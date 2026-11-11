"""Tests for the HCRL loader using hand-written sample lines (synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from canids.ingestion.hcrl_loader import (
    load_hcrl_dir,
    load_hcrl_file,
    parse_hcrl_csv_line,
    parse_hcrl_txt_line,
)
from canids.ingestion.schema import ParseError, validate_frame_df


# ------------------------------------------------------------------ CSV lines
def test_csv_full_dlc_regular() -> None:
    (ts, cid, dlc, payload), flag = parse_hcrl_csv_line(
        "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
    )
    assert ts == pytest.approx(1478198376.389427)
    assert (cid, dlc, flag) == (0x316, 8, "R")
    assert payload == (0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6F)


def test_csv_short_dlc_flag_shifted_left() -> None:
    (_, cid, dlc, payload), flag = parse_hcrl_csv_line("1478198376.389636,0080,2,00,17,T")
    assert (cid, dlc, flag) == (0x80, 2, "T")
    assert payload == (0x00, 0x17, -1, -1, -1, -1, -1, -1)


def test_csv_dlc_zero() -> None:
    (_, _, dlc, payload), flag = parse_hcrl_csv_line("1.0,0000,0,R")
    assert dlc == 0 and flag == "R" and payload == (-1,) * 8


def test_csv_custom_missing_value() -> None:
    (_, _, _, payload), _ = parse_hcrl_csv_line("1.0,0080,1,aa,R", missing_byte=0)
    assert payload == (0xAA, 0, 0, 0, 0, 0, 0, 0)


def test_csv_trailing_empty_fields_tolerated() -> None:
    (_, _, dlc, _), flag = parse_hcrl_csv_line("1.0,0080,2,00,17,T,,,,,,")
    assert dlc == 2 and flag == "T"


@pytest.mark.parametrize(
    "line, reason",
    [
        ("", "empty_line"),
        ("   \n", "empty_line"),
        ("Timestamp,CAN ID,DLC,DATA0,Flag", "bad_timestamp"),
        ("abc,0316,8,05,21,68,09,21,21,00,6f,R", "bad_timestamp"),
        ("1.0,ZZZZ,8,05,21,68,09,21,21,00,6f,R", "bad_id"),
        ("1.0,0316,9,05,21,68,09,21,21,00,6f,R", "bad_dlc"),
        ("1.0,0316,x,05,R", "bad_dlc"),
        ("1.0,0316,8,05,21,68,G9,21,21,00,6f,R", "bad_hex"),
        ("1.0,0316,8,05,21,68,09,21,21,00,6f,X", "bad_flag"),
        ("1.0,0316,8,05,21,68", "too_few_fields"),
        ("1.0,0316", "too_few_fields"),
        ("1.0,0080,2,00,17,T,99", "extra_fields"),
    ],
)
def test_csv_malformed(line: str, reason: str) -> None:
    with pytest.raises(ParseError) as err:
        parse_hcrl_csv_line(line)
    assert err.value.reason == reason


# ------------------------------------------------------------------ TXT lines
TXT_OK = "Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2"


def test_txt_line() -> None:
    ts, cid, dlc, payload = parse_hcrl_txt_line(TXT_OK)
    assert ts == pytest.approx(1479121434.850202)
    assert (cid, dlc) == (0x350, 8)
    assert payload == (0x05, 0x28, 0x84, 0x66, 0x6D, 0x00, 0x00, 0xA2)


def test_txt_short_dlc() -> None:
    _, _, dlc, payload = parse_hcrl_txt_line("Timestamp: 1.5 ID: 0080 000 DLC: 2 00 17")
    assert dlc == 2 and payload[:3] == (0, 0x17, -1)


@pytest.mark.parametrize(
    "line, reason",
    [
        ("", "empty_line"),
        ("garbage line", "bad_format"),
        ("Timestamp: 1.0 ID: 0350 000 DLC: 8 05 28", "dlc_mismatch"),
        ("Timestamp: 1.0 ID: 0350 000 DLC: 2 05 QQ", "bad_hex"),
    ],
)
def test_txt_malformed(line: str, reason: str) -> None:
    with pytest.raises(ParseError) as err:
        parse_hcrl_txt_line(line)
    assert err.value.reason == reason


# ------------------------------------------------------------------ files
SAMPLE_CSV = """1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
1478198376.389636,0000,8,00,00,00,00,00,00,00,00,T

1478198376.389900,0080,2,00,17,R
1478198376.390000,0316,8,05,21,68,ZZ,21,21,00,6f,R
1478198376.390100,0000,8,00,00,00,00,00,00,00,00,T
"""


def test_load_csv_file_labels_and_drop_counts(tmp_path: Path) -> None:
    f = tmp_path / "DoS_dataset.csv"
    f.write_text(SAMPLE_CSV, encoding="utf-8")
    df, report = load_hcrl_file(f, label="DoS", attack_type="dos")
    validate_frame_df(df)
    assert report.rows_read == 6
    assert report.dropped == {"empty_line": 1, "bad_hex": 1}
    assert report.kept_per_label == {"Normal": 2, "DoS": 2}
    assert set(df.loc[df["label"] == "DoS", "attack_type"]) == {"dos"}
    assert set(df["source_dataset"]) == {"hcrl"}
    assert set(df["source_file"]) == {"DoS_dataset.csv"}


def test_load_txt_file_all_normal(tmp_path: Path) -> None:
    f = tmp_path / "normal_run_data.txt"
    f.write_text(TXT_OK + "\n" + TXT_OK.replace("850202", "851000") + "\n", encoding="utf-8")
    df, report = load_hcrl_file(f, label="Normal", attack_type="none")
    assert report.kept_per_label == {"Normal": 2}
    assert len(df) == 2


def test_max_rows(tmp_path: Path) -> None:
    f = tmp_path / "DoS_dataset.csv"
    f.write_text(SAMPLE_CSV, encoding="utf-8")
    df, report = load_hcrl_file(f, label="DoS", attack_type="dos", max_rows=2)
    assert report.rows_read == 2 and len(df) == 2


def test_invalid_label_rejected(tmp_path: Path) -> None:
    f = tmp_path / "x.csv"
    f.write_text(SAMPLE_CSV, encoding="utf-8")
    with pytest.raises(ValueError):
        load_hcrl_file(f, label="Hack", attack_type="x")


def test_load_dir_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "DoS_dataset.csv").write_text(SAMPLE_CSV, encoding="utf-8")
    files = {
        "DoS_dataset.csv": {"label": "DoS", "attack_type": "dos"},
        "Fuzzy_dataset.csv": {"label": "Fuzzy", "attack_type": "fuzzy"},
    }
    results = list(load_hcrl_dir(tmp_path, files))
    assert results[0][0] is not None and not results[0][1].skipped
    assert results[1][0] is None and results[1][1].skipped
