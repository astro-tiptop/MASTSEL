"""
Tests for the LO performance changes: memoized sympy substitutions
(``_cached_subs``), the once-per-expression noise lambdify (``_noiseLambda``)
and the lazy import of the formulary / astropy.modeling.
"""

import configparser
import os
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from mastsel.mavisLO import (MavisLO, parse_config_value, _cached_subs,
                             lambdifyByName)
from mastsel.mavisFormulas import getMavisFormulary

INI_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ini",
                        "mavisParamsTests.ini")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Largest relative difference seen in the noise-lambda comparison (reported at the end)
_max_rel_diff = [0.0]


def _load_config() -> dict:
    """Load the test ini as a nested dict, with the same value parsing as MavisLO."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(INI_PATH)
    return {sec: {k: parse_config_value(v) for k, v in parser[sec].items()}
            for sec in parser.sections()}


def _make_lo() -> MavisLO:
    cfg = _load_config()
    cfg.setdefault("COMPUTATION", {})
    cfg["COMPUTATION"].update(platform="CPU")
    return MavisLO(config_dict=cfg)


# ---------------------------------------------------------------- _cached_subs

def _turb_subs(F) -> dict:
    sm = F.symbol_map
    return {sm['V']: 10.0, sm['R']: 4.0, sm['r_0']: 0.15, sm['L_0']: 25.0,
            sm['k_y_min']: 1e-4, sm['k_y_max']: 1000}


def test_cached_subs_matches_plain_subs_formulary():
    F = getMavisFormulary()
    d = _turb_subs(F)
    for name in ('turbPSDTip', 'turbPSDTilt'):
        assert _cached_subs(F[name], d) == F[name].subs(d)


def test_cached_subs_matches_plain_subs_simple():
    x, y, z = sp.symbols('x y z')
    expr = x * sp.sin(y) + z**2
    d = {x: 3.5, y: 0.25}
    assert _cached_subs(expr, d) == expr.subs(d)


def test_cached_subs_cache_hit_returns_same_object():
    x, y = sp.symbols('hit_x hit_y')
    expr = x**2 + y
    d = {x: 2.0, y: 7.0}
    first = _cached_subs(expr, d)
    assert _cached_subs(expr, dict(d)) is first
    F = getMavisFormulary()
    r1 = _cached_subs(F['turbPSDTip'], _turb_subs(F))
    assert _cached_subs(F['turbPSDTip'], _turb_subs(F)) is r1


def test_cached_subs_int_then_float():
    x, y = sp.symbols('if_x if_y')
    expr = x * y
    r_int = _cached_subs(expr, {y: 120})
    r_float = _cached_subs(expr, {y: 120.0})
    assert not r_int.atoms(sp.Float) and r_int.atoms(sp.Integer)
    assert r_float.atoms(sp.Float)
    assert r_int != r_float or r_int.atoms(sp.Float) != r_float.atoms(sp.Float)


def test_cached_subs_float_then_int():
    x, y = sp.symbols('fi_x fi_y')
    expr = x * y
    r_float = _cached_subs(expr, {y: 120.0})
    r_int = _cached_subs(expr, {y: 120})
    assert r_float.atoms(sp.Float)
    assert not r_int.atoms(sp.Float) and r_int.atoms(sp.Integer)


def test_cached_subs_unhashable_value_falls_back():
    x, y = sp.symbols('uh_x uh_y')
    expr = x + y
    d = {y: [1, 2]}
    try:
        expected = expr.subs(d)
    except Exception as exc:
        with pytest.raises(type(exc)):
            _cached_subs(expr, d)
    else:
        assert _cached_subs(expr, d) == expected


# ----------------------------------------------------------------- noise lambda

@pytest.fixture(scope="module")
def lo() -> MavisLO:
    obj = _make_lo()
    obj.configLOFreq(500.0)  # defines fTipS_LO / fTipS (per-NGS configuration)
    return obj


def _arrays():
    rng = np.random.default_rng(0)
    n_g, n_f = 8, 500
    g = np.logspace(-4, np.log10(0.5), n_g).reshape(n_g, 1)
    f = np.linspace(0.5, 250, n_f).reshape(1, n_f)
    psd = 10 ** rng.uniform(-3, 0, size=(1, n_f))
    return g, f, psd


@pytest.mark.parametrize("attr", ["fTipS_LO", "fTipS"])
@pytest.mark.parametrize("sigma2", [1e-6, 1e-3, 0.1])
def test_noise_lambda_matches_subs_lambdify(lo, attr, sigma2):
    if not hasattr(lo, attr):
        pytest.skip(f"{attr} not defined on this MavisLO")
    expr = getattr(lo, attr)
    noise = lo.MavisFormulas.symbol_map['phi^noise_Tip']
    names = ['g^Tip_0', 'f', 'phi^wind_Tip']
    ref_fn = lambdifyByName(expr.subs({noise: sigma2}), names, lo.platformlib)
    new_fn = lo._noiseLambda(expr, names, 'phi^noise_Tip', sigma2)
    g, f, psd = _arrays()
    ref = np.asarray(ref_fn(g, f, psd))
    new = np.asarray(new_fn(g, f, psd))
    assert ref.shape == new.shape
    rel = float(np.max(np.abs(new - ref) / np.maximum(np.abs(ref), 1e-300)))
    _max_rel_diff[0] = max(_max_rel_diff[0], rel)
    print(f"max rel diff {attr} sigma2={sigma2}: {rel:.3e}")
    np.testing.assert_allclose(new, ref, rtol=1e-12, atol=0)


def test_noise_lambda_reuses_lambdify(lo):
    from mastsel.mavisLO import _lambdify_cached
    names = ['g^Tip_0', 'f', 'phi^wind_Tip']
    lo._noiseLambda(lo.fTipS_LO, names, 'phi^noise_Tip', 1e-3)
    before = _lambdify_cached.cache_info()
    lo._noiseLambda(lo.fTipS_LO, names, 'phi^noise_Tip', 0.1)
    after = _lambdify_cached.cache_info()
    assert after.hits == before.hits + 1 and after.misses == before.misses


# ------------------------------------------------------------------ lazy import

_SUBPROCESS_CODE = """
import sys
import mastsel
assert mastsel.__file__.startswith({root!r}), mastsel.__file__
import mastsel.mavisFormulas as mf
assert mf.getMavisFormulary.cache_info().currsize == 0, 'formulary built at import'
assert 'astropy.modeling' not in sys.modules, 'astropy.modeling imported eagerly'
obj = mf._mavisFormulas
assert mf.getMavisFormulary.cache_info().currsize == 1
assert obj is mf.getMavisFormulary()
assert type(obj).__name__ == 'Formulary', type(obj).__name__
print('OK')
"""


def test_import_is_lazy_in_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    res = subprocess.run([sys.executable, "-c", _SUBPROCESS_CODE.format(root=REPO_ROOT)],
                         env=env, cwd="/tmp", capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.strip().endswith("OK")


def test_report_max_rel_diff():
    print(f"overall max relative difference (noise lambda): {_max_rel_diff[0]:.3e}")
