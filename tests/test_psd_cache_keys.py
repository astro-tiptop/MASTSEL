"""
Regression tests for the cross-instance PSD cache in MavisLO.

``_compute_turb_psds_cached`` and ``_compute_focus_psds_cached`` use
``method_lru_cache``, a cache shared by all instances that ignores ``self``.
The cache key must therefore contain every instance attribute the result
depends on: the number of integration points ([COMPUTATION]
integralDiscretization2), ``max_freq_turb`` and the dtype ([COMPUTATION]
precision). Before the fix, a second instance with a different
integralDiscretization2 or precision silently received the PSDs of the first.
"""

import configparser
import os

import numpy as np
import pytest

from mastsel.mavisLO import MavisLO, parse_config_value

# MASTSEL-format ini (the TIPTOP MAVIS.ini uses a different key layout)
INI_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ini",
                        "mavisParamsTests.ini")

FMIN, FMAX, FREQ_SAMPLES = 1.0, 100.0, 20


def _to_host(a) -> np.ndarray:
    """Return a numpy array from a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def _load_config() -> dict:
    """Load the test ini as a nested dict, with the same value parsing as MavisLO."""
    parser = configparser.ConfigParser()
    parser.optionxform = str  # keep key case: MavisLO looks keys up case-sensitively
    parser.read(INI_PATH)
    return {sec: {k: parse_config_value(v) for k, v in parser[sec].items()}
            for sec in parser.sections()}


def _make_lo(**computation) -> MavisLO:
    """Build a MavisLO from the test ini with [COMPUTATION] overrides."""
    cfg = _load_config()
    cfg.setdefault("COMPUTATION", {})
    cfg["COMPUTATION"].update(platform="CPU", **computation)
    return MavisLO(config_dict=cfg)


def _rel_diff(a, b) -> float:
    a, b = _to_host(a), _to_host(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


@pytest.fixture(autouse=True)
def clean_caches():
    MavisLO._compute_turb_psds_cached.cache_clear()
    MavisLO._compute_focus_psds_cached.cache_clear()
    yield
    MavisLO._compute_turb_psds_cached.cache_clear()
    MavisLO._compute_focus_psds_cached.cache_clear()


def _check_no_cross_talk(compute, clear, lo_a, lo_b, rtol, min_diff):
    """
    Compute with A, then B (cached run), then fresh B after clearing the cache.
    B must not inherit A's result, and A and fresh B must differ measurably.
    """
    res_a = compute(lo_a)
    res_b = compute(lo_b)
    clear()
    res_b_fresh = compute(lo_b)
    for x_b, x_fresh in zip(res_b, res_b_fresh):
        assert _to_host(x_b).dtype == _to_host(x_fresh).dtype
        np.testing.assert_allclose(_to_host(x_b), _to_host(x_fresh), rtol=rtol, atol=0)
    # at least one output must be sensitive to the changed setting
    assert max(_rel_diff(x_a, x_f) for x_a, x_f in zip(res_a, res_b_fresh)) > min_diff


def _turb(lo):
    return lo.computeTurbPSDs(FMIN, FMAX, FREQ_SAMPLES)


def _focus(lo):
    return lo.computeFocusPSDs(FMIN, FMAX, FREQ_SAMPLES)


def test_turb_psd_cache_depends_on_integration_points():
    lo_a = _make_lo(integralDiscretization2=1000, precision="double")
    lo_b = _make_lo(integralDiscretization2=100, precision="double")
    assert lo_a.psdIntegrationPoints != lo_b.psdIntegrationPoints
    _check_no_cross_talk(_turb, MavisLO._compute_turb_psds_cached.cache_clear,
                         lo_a, lo_b, rtol=1e-12, min_diff=1e-6)


def test_turb_psd_cache_depends_on_precision():
    lo_a = _make_lo(precision="double")
    lo_b = _make_lo(precision="single")
    _check_no_cross_talk(_turb, MavisLO._compute_turb_psds_cached.cache_clear,
                         lo_a, lo_b, rtol=1e-12, min_diff=0.0)


def test_focus_psd_cache_depends_on_integration_points():
    lo_a = _make_lo(integralDiscretization2=1000, precision="double")
    lo_b = _make_lo(integralDiscretization2=100, precision="double")
    _check_no_cross_talk(_focus, MavisLO._compute_focus_psds_cached.cache_clear,
                         lo_a, lo_b, rtol=1e-12, min_diff=1e-6)


def test_focus_psd_cache_depends_on_precision():
    lo_a = _make_lo(precision="double")
    lo_b = _make_lo(precision="single")
    # outputs stay float64 even in single precision; A and B differ only at ~1e-6,
    # so the cached-vs-fresh comparison must be tight (rtol 1e-12) to discriminate
    _check_no_cross_talk(_focus, MavisLO._compute_focus_psds_cached.cache_clear,
                         lo_a, lo_b, rtol=1e-12, min_diff=0.0)
