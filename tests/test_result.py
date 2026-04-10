"""Unit tests for Result accumulation in szcore-evaluation.

All tests use synthetic numpy masks — no file I/O required.
"""

import math

import numpy as np
import pytest

from timescoring.annotations import Annotation
from timescoring import scoring
from szcore_evaluation.evaluate import Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_scoring(
    ref_mask: np.ndarray, hyp_mask: np.ndarray, fs: int = 1
) -> scoring.SampleScoring:
    """Return a SampleScoring from two boolean numpy masks."""
    return scoring.SampleScoring(
        Annotation(ref_mask, fs),
        Annotation(hyp_mask, fs),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_mask():
    """10-sample recording: 4 reference-positive samples (indices 2-5)."""
    return np.array([False, False, True, True, True, True, False, False, False, False])


@pytest.fixture
def hyp_mask(ref_mask):
    """Hypothesis: 3 positive samples — 2 TP (indices 2-3), 1 FP (index 6).

    TP at indices 2-3 (where ref_mask is True).
    FP at index 6 (where ref_mask is False).
    FN at indices 4-5 (where ref_mask is True but hyp_mask is False).
    """
    hyp = ref_mask.copy()
    for i in [6, 4, 5]:
        hyp[i] = ~hyp[i]  # Flip the value at these indices
    return hyp


@pytest.fixture
def perfect_hyp_mask(ref_mask):
    """Perfect hypothesis: identical to reference."""
    return ref_mask.copy()


@pytest.fixture
def sample_score(ref_mask, hyp_mask):
    """SampleScoring with known values: tp=2, fp=1, refTrue=4, numSamples=10."""
    return make_sample_scoring(ref_mask, hyp_mask)


@pytest.fixture
def perfect_score(ref_mask, perfect_hyp_mask):
    """SampleScoring with perfect detection: tp=4, fp=0, refTrue=4."""
    return make_sample_scoring(ref_mask, perfect_hyp_mask)


# ---------------------------------------------------------------------------
# Test Result zero initialisation
# ---------------------------------------------------------------------------

def test_result_zero_init():
    """Result() with no argument initialises all attributes to zero."""
    r = Result()
    assert r.fs == 0
    assert r.duration == 0
    assert r.numSamples == 0
    assert r.tp == 0
    assert r.fp == 0
    assert r.refTrue == 0


# ---------------------------------------------------------------------------
# Test Result initialisation from SampleScoring
# ---------------------------------------------------------------------------

def test_result_from_sample_scoring(sample_score):
    """Result(score) copies attributes correctly from SampleScoring."""
    r = Result(sample_score)
    assert r.tp == 2        # 2 TP: indices 2 and 3
    assert r.fp == 1        # 1 FP: index 6
    assert r.refTrue == 4   # 4 reference-positive samples
    assert r.numSamples == 10
    assert r.fs == 1
    assert r.duration == pytest.approx(10.0)  # 10 samples at 1 Hz


# ---------------------------------------------------------------------------
# Test Result.__add__
# ---------------------------------------------------------------------------

def test_result_add_sums_attributes(sample_score, perfect_score):
    """Result + Result sums tp, fp, refTrue, numSamples, duration."""
    r1 = Result(sample_score)   # tp=2, fp=1, refTrue=4, numSamples=10
    r2 = Result(perfect_score)  # tp=4, fp=0, refTrue=4, numSamples=10

    combined = r1 + r2

    assert combined.tp == 6
    assert combined.fp == 1
    assert combined.refTrue == 8
    assert combined.numSamples == 20
    assert combined.duration == pytest.approx(20.0)


def test_result_add_returns_new_object(sample_score, perfect_score):
    """Result + Result returns a new Result, leaving operands unchanged."""
    r1 = Result(sample_score)
    r2 = Result(perfect_score)
    combined = r1 + r2

    # Operands unchanged
    assert r1.tp == 2
    assert r2.tp == 4
    # Result is a new object
    assert combined is not r1
    assert combined is not r2


def test_result_add_zero_identity(sample_score):
    """Result() + Result(score) equals Result(score) in all attributes."""
    r_zero = Result()
    r_score = Result(sample_score)
    combined = r_zero + r_score

    assert combined.tp == r_score.tp
    assert combined.fp == r_score.fp
    assert combined.refTrue == r_score.refTrue
    assert combined.numSamples == r_score.numSamples


# ---------------------------------------------------------------------------
# Test Result.__iadd__
# ---------------------------------------------------------------------------

def test_result_iadd_sums_attributes(sample_score, perfect_score):
    """Result += Result sums attributes in-place."""
    r = Result(sample_score)    # tp=2, fp=1, refTrue=4, numSamples=10
    r += Result(perfect_score)  # tp=4, fp=0, refTrue=4, numSamples=10

    assert r.tp == 6
    assert r.fp == 1
    assert r.refTrue == 8
    assert r.numSamples == 20
    assert r.duration == pytest.approx(20.0)


def test_result_iadd_matches_add(sample_score, perfect_score):
    """Result += other produces the same values as Result + other."""
    r1_add = Result(sample_score) + Result(perfect_score)

    r1_iadd = Result(sample_score)
    r1_iadd += Result(perfect_score)

    assert r1_iadd.tp == r1_add.tp
    assert r1_iadd.fp == r1_add.fp
    assert r1_iadd.refTrue == r1_add.refTrue
    assert r1_iadd.numSamples == r1_add.numSamples
    assert r1_iadd.duration == pytest.approx(r1_add.duration)


def test_result_iadd_from_zero(sample_score):
    """Accumulating into a zero Result via += yields same values as Result(score)."""
    r = Result()
    r += Result(sample_score)

    expected = Result(sample_score)
    assert r.tp == expected.tp
    assert r.fp == expected.fp
    assert r.refTrue == expected.refTrue
    assert r.numSamples == expected.numSamples


# ---------------------------------------------------------------------------
# Test Result.computeScores()
# ---------------------------------------------------------------------------

def test_compute_scores_known_values(sample_score):
    """computeScores() produces correct metrics for known tp/fp/refTrue."""
    # tp=2, fp=1, refTrue=4, numSamples=10, fs=1
    r = Result(sample_score)
    r.computeScores()

    assert r.sensitivity == pytest.approx(2 / 4)       # 0.5
    assert r.precision == pytest.approx(2 / (2 + 1))   # 0.667
    # f1 = 2*tp / (2*tp + fp + fn) = 4 / (4 + 1 + 2) = 4/7
    assert r.f1 == pytest.approx(4 / 7)
    # fpRate = fp / (numSamples / fs / 3600 / 24) = 1 / (10/1/3600/24)
    expected_fpRate = 1 / (10 / 1 / 3600 / 24)
    assert r.fpRate == pytest.approx(expected_fpRate)


def test_compute_scores_perfect_detection(perfect_score):
    """Perfect detection: sensitivity=1.0, precision=1.0, f1=1.0."""
    r = Result(perfect_score)
    r.computeScores()

    assert r.sensitivity == pytest.approx(1.0)
    assert r.precision == pytest.approx(1.0)
    assert r.f1 == pytest.approx(1.0)
    assert r.fpRate == pytest.approx(0.0)


def test_compute_scores_after_accumulation(sample_score, perfect_score):
    """computeScores() works correctly on an accumulated Result."""
    r = Result(sample_score) + Result(perfect_score)
    # tp=6, fp=1, refTrue=8, numSamples=20
    r.computeScores()

    assert r.sensitivity == pytest.approx(6 / 8)
    assert r.precision == pytest.approx(6 / 7)


def test_compute_scores_no_reference():
    """sensitivity is nan when refTrue == 0 (no reference events)."""
    ref_mask = np.zeros(10, dtype=bool)
    hyp_mask = np.zeros(10, dtype=bool)
    score = make_sample_scoring(ref_mask, hyp_mask)
    r = Result(score)
    r.computeScores()

    assert math.isnan(r.sensitivity)


def test_sample_scoring_rejects_mismatched_lengths():
    """SampleScoring raises ValueError when ref and hyp masks differ in length."""
    ref_mask = np.zeros(10, dtype=bool)
    hyp_mask = np.zeros(8, dtype=bool)
    with pytest.raises(ValueError):
        make_sample_scoring(ref_mask, hyp_mask)
