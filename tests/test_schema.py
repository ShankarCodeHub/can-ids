"""Tests for the unified frame schema."""

from __future__ import annotations

import pandas as pd
import pytest

from canids.ingestion.schema import (
    COLUMNS,
    MISSING_BYTE,
    CANFrame,
    FrameAccumulator,
    ParseError,
    SchemaError,
    empty_frame_df,
    frames_to_dataframe,
    pad_payload,
    parse_can_id,
    parse_hex_byte,
    validate_frame_df,
)


def test_parse_hex_byte() -> None:
    assert parse_hex_byte("0f") == 15
    assert parse_hex_byte("FF") == 255
    assert parse_hex_byte("a") == 10
    for bad in ("zz", "100", "", "-1"):
        with pytest.raises(ParseError) as err:
            parse_hex_byte(bad)
        assert err.value.reason == "bad_hex"


def test_parse_can_id() -> None:
    assert parse_can_id("0316") == 0x316
    assert parse_can_id("0x43f") == 0x43F
    assert parse_can_id("18FEF100") == 0x18FEF100
    for bad in ("xyz", "", "3FFFFFFF"):
        with pytest.raises(ParseError) as err:
            parse_can_id(bad)
        assert err.value.reason == "bad_id"


def test_pad_payload() -> None:
    assert pad_payload([1, 2]) == (1, 2, -1, -1, -1, -1, -1, -1)
    assert pad_payload([], fill=0) == (0,) * 8
    with pytest.raises(ParseError):
        pad_payload(list(range(9)))


def _frame(**kw: object) -> CANFrame:
    base = dict(timestamp=1.0, can_id=0x316, dlc=2, payload=pad_payload([5, 6]))
    base.update(kw)
    return CANFrame(**base)  # type: ignore[arg-type]


def test_canframe_roundtrip_and_validation() -> None:
    f = _frame()
    assert f.data == b"\x05\x06"
    df = frames_to_dataframe([f, _frame(timestamp=2.0)])
    assert list(df.columns) == COLUMNS
    validate_frame_df(df)
    assert df.loc[0, "b2"] == MISSING_BYTE


def test_empty_df_is_valid() -> None:
    validate_frame_df(empty_frame_df())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.assign(dlc=pd.Series([9, 2], dtype="int8")), "dlc"),
        (lambda d: d.assign(b0=pd.Series([-1, 5], dtype="int16")), "missing byte inside"),
        (lambda d: d.assign(b5=pd.Series([7, -1], dtype="int16")), "beyond the DLC"),
        (lambda d: d.assign(label=["Normal", "Hack"]), "unknown labels"),
        (lambda d: d.drop(columns=["b7"]), "columns"),
    ],
)
def test_validate_rejects_bad_frames(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    df = frames_to_dataframe([_frame(), _frame(timestamp=2.0)])
    with pytest.raises(SchemaError, match=message):
        validate_frame_df(mutate(df))


def test_accumulator_chunks_and_unifies_categories() -> None:
    acc = FrameAccumulator("hcrl", "x.csv", chunk_size=2)
    for i, label in enumerate(["Normal", "Normal", "DoS", "Normal", "DoS"]):
        acc.append(float(i), 0, 0, (MISSING_BYTE,) * 8, label, "dos" if label == "DoS" else "none")
    assert len(acc) == 5
    df = acc.to_dataframe()
    validate_frame_df(df)
    assert df["label"].dtype.name == "category"
    assert df["label"].value_counts().to_dict() == {"Normal": 3, "DoS": 2}
    assert df["timestamp"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
