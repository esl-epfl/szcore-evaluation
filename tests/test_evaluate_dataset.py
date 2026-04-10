"""Integration tests for evaluate_dataset() in szcore-evaluation.

All tests use synthetic HED-SCORE TSV files written to pytest's tmp_path.
No real EEG data or network access required.
"""

from pathlib import Path

import pytest

from szcore_evaluation.evaluate import evaluate_dataset

METRICS = ["sensitivity", "precision", "f1", "fpRate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tsv(path: Path, events: list, recording_duration: int = 60) -> None:
    """Write a minimal HED-SCORE TSV file.

    Args:
        path: destination file path (parent dirs are created automatically)
        events: list of (onset, duration) tuples for seizure events;
                pass [] for a background-only (seizure-free) recording
        recording_duration: total recording length in seconds
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "onset\tduration\teventType\tconfidence\tchannels\tdateTime\trecordingDuration\n"
    rows = []
    if events:
        for onset, duration in events:
            rows.append(
                f"{onset}\t{duration}\tsz\t1.0\tn/a\t2000-01-01 00:00:00\t{recording_duration}\n"
            )
    else:
        rows.append(
            f"0\t{recording_duration}\tbckg\tn/a\tn/a\t2000-01-01 00:00:00\t{recording_duration}\n"
        )
    path.write_text(header + "".join(rows))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_evaluate_dataset_output_keys(tmp_path):
    """Output contains sample_results and event_results with all required keys."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [(10, 5)])
    make_tsv(tmp_path / "hyp" / "sub-001" / "rec.tsv", [(10, 5)])

    result = evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json"
    )

    assert "sample_results" in result
    assert "event_results" in result
    for section in ("sample_results", "event_results"):
        for key in METRICS:
            assert key in result[section], f"missing {key} in {section}"
            assert f"{key}_std" in result[section], f"missing {key}_std in {section}"
    assert (tmp_path / "out.json").exists()


def test_evaluate_dataset_perfect_detection(tmp_path):
    """Identical ref and hyp yield sensitivity=1.0 and precision=1.0."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [(10, 5)])
    make_tsv(tmp_path / "hyp" / "sub-001" / "rec.tsv", [(10, 5)])

    result = evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json"
    )

    assert result["sample_results"]["sensitivity"] == pytest.approx(1.0)
    assert result["sample_results"]["precision"] == pytest.approx(1.0)


def test_evaluate_dataset_missing_hyp(tmp_path):
    """Missing hypothesis TSV is treated as all-zero predictions; JSON is written."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [(10, 5)])
    (tmp_path / "hyp").mkdir()  # hyp dir exists but contains no files

    result = evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json"
    )

    assert (tmp_path / "out.json").exists()
    assert result["sample_results"]["sensitivity"] == pytest.approx(0.0)


def test_evaluate_dataset_empty_reference(tmp_path):
    """Seizure-free reference (background only) does not raise an exception."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [])  # no seizures
    make_tsv(tmp_path / "hyp" / "sub-001" / "rec.tsv", [])

    evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json"
    )

    assert (tmp_path / "out.json").exists()


def test_evaluate_dataset_sensitivity_std(tmp_path):
    """avg_per_subject=True with 2 subjects: sensitivity_std reflects per-subject variance."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [(10, 5)])
    make_tsv(tmp_path / "ref" / "sub-002" / "rec.tsv", [(10, 5)])
    make_tsv(tmp_path / "hyp" / "sub-001" / "rec.tsv", [(10, 5)])  # perfect match
    make_tsv(tmp_path / "hyp" / "sub-002" / "rec.tsv", [])          # miss all

    result = evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json",
        avg_per_subject=True,
    )

    assert "sensitivity_std" in result["sample_results"]
    assert result["sample_results"]["sensitivity_std"] > 0


def test_evaluate_dataset_cumulated(tmp_path):
    """avg_per_subject=False writes cumulated results without error."""
    make_tsv(tmp_path / "ref" / "sub-001" / "rec.tsv", [(10, 5)])
    make_tsv(tmp_path / "hyp" / "sub-001" / "rec.tsv", [(10, 5)])

    result = evaluate_dataset(
        tmp_path / "ref", tmp_path / "hyp", tmp_path / "out.json",
        avg_per_subject=False,
    )

    assert (tmp_path / "out.json").exists()
    for key in METRICS:
        assert key in result["sample_results"]
