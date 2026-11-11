"""Tests for per-file frame cleaning."""

from __future__ import annotations

import pandas as pd
import pytest

from canids.ingestion.schema import CANFrame, frames_to_dataframe, pad_payload, validate_frame_df
from canids.preprocessing.cleaning import clean_frames


def _df(rows: list[tuple[float, int, list[int]]], source: str = "f.log") -> pd.DataFrame:
    return frames_to_dataframe(
        CANFrame(t, cid, len(p), pad_payload(p), source_file=source, label="Normal")
        for t, cid, p in rows
    )


def test_sorts_and_drops_exact_duplicates() -> None:
    df = _df(
        [
            (2.0, 0x10, [1]),
            (1.0, 0x10, [1]),
            (1.0, 0x10, [1]),  # exact duplicate -> dropped
            (1.0, 0x20, [1]),  # same time, different ID -> kept
            (3.0, 0x10, [1]),  # same ID+payload, different time -> kept (periodic traffic)
        ]
    )
    out, rep = clean_frames(df)
    validate_frame_df(out)
    assert out["timestamp"].tolist() == [1.0, 1.0, 2.0, 3.0]
    assert out["can_id"].tolist()[:2] == [0x10, 0x20]  # stable order kept for equal timestamps
    assert rep.duplicates_dropped == 1 and rep.was_unsorted and rep.out_of_order_frames == 1
    assert (rep.rows_in, rep.rows_out) == (5, 4)


def test_already_clean_input_untouched() -> None:
    df = _df([(0.0, 1, [1]), (0.5, 2, [2])])
    out, rep = clean_frames(df)
    assert not rep.was_unsorted and rep.duplicates_dropped == 0
    pd.testing.assert_frame_equal(out, df)


def test_rejects_mixed_files() -> None:
    df = pd.concat([_df([(0.0, 1, [1])], "a.log"), _df([(0.0, 1, [1])], "b.log")])
    with pytest.raises(ValueError):
        clean_frames(df)
