import numpy as np
import pytest
import sympy as sp

cp = pytest.importorskip('cupy')

try:
    cp.cuda.runtime.getDeviceCount()
except Exception:
    pytest.skip('no CUDA device', allow_module_level=True)

import mastsel
from mastsel.mavisLO import _FusedOrPlain, _lambdify_cached

if not mastsel.gpuEnabled:
    pytest.skip('MASTSEL GPU disabled', allow_module_level=True)


def _x():
    return cp.linspace(-5.0, 5.0, 1000)


def test_fusable_function_uses_fused_after_first_call():
    f = _FusedOrPlain(lambda x: cp.exp(1j * 0.008 * cp.pi * x) / (1 + x**2))
    x = _x()
    ref = np.exp(1j * 0.008 * np.pi * cp.asnumpy(x)) / (1 + cp.asnumpy(x)**2)

    first = f(x)
    assert f.use_fused is True
    assert cp.array_equal(first, f.plain(x))  # first call returns the plain result

    second = f(x)
    np.testing.assert_allclose(cp.asnumpy(second), ref, rtol=1e-12, atol=0)


def test_constant_function_falls_back_to_plain():
    # cp.tan(0.3) alone inside cp.fuse is evaluated in float16
    f = _FusedOrPlain(lambda x: x * cp.tan(0.3))
    x = _x()
    ref = cp.asnumpy(x) * np.tan(0.3)
    for _ in range(3):
        np.testing.assert_allclose(cp.asnumpy(f(x)), ref, rtol=1e-14, atol=0)
        assert f.use_fused is False


def test_unfusable_function_falls_back_without_raising():
    def g(x):
        if x.sum() > 0:  # Python branch on array values: not fusable
            return x * 2.0
        return x

    f = _FusedOrPlain(g)
    x = _x() + 10.0
    for _ in range(2):
        np.testing.assert_allclose(cp.asnumpy(f(x)), cp.asnumpy(x) * 2.0, rtol=1e-14)
    assert f.use_fused is False


def test_lambdify_cached_gpu_and_cpu():
    f_sym, g_sym = sp.symbols('f g')
    expr = sp.exp(sp.I * f_sym) * g_sym
    fv = np.linspace(0.1, 3.0, 500)
    ref = np.exp(1j * fv) * 2.5

    fgpu = _lambdify_cached(expr, ('f', 'g'), True)
    assert isinstance(fgpu, _FusedOrPlain)
    for _ in range(2):
        np.testing.assert_allclose(cp.asnumpy(fgpu(cp.asarray(fv), 2.5)), ref, rtol=1e-12, atol=0)

    fcpu = _lambdify_cached(expr, ('f', 'g'), False)
    assert not isinstance(fcpu, _FusedOrPlain)
    np.testing.assert_allclose(fcpu(fv, 2.5), ref, rtol=1e-12, atol=0)
