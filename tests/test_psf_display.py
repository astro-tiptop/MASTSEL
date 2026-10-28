import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mastsel.mavisUtilities import tiledDisplay, standardPsfPlot


def _psf(value, size=4):
    """Minimal PSF-like object with a constant `.sampling` image."""
    return SimpleNamespace(sampling=np.full((size, size), float(value)))


def _grid_by_position(fig):
    """Map (row from top, col) -> axes, using the subplot spec."""
    out = {}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        out[(spec.rowspan.start, spec.colspan.start)] = ax
    return out


class TestPsfDisplay(unittest.TestCase):
    def setUp(self):
        self._show = plt.show
        plt.show = lambda *a, **k: None
        plt.close('all')

    def tearDown(self):
        plt.show = self._show
        plt.close('all')

    def _value_at(self, axes, row, col):
        # log(|v| + 1e-20) of a constant image
        return np.exp(axes[(row, col)].images[0].get_array()[0, 0])

    def test_placement_by_coords(self):
        # 3 x values, 2 y values, shuffled; PSF k has value k+1
        coords = np.array([[10, -5], [-10, 5], [0, 5], [10, 5], [-10, -5], [0, -5]], float)
        results = [_psf(k + 1) for k in range(len(coords))]
        tiledDisplay(results, coords=coords)
        fig = plt.gcf()
        self.assertEqual(len(fig.axes), 6)
        axes = _grid_by_position(fig)
        self.assertEqual(max(r for r, _ in axes) + 1, 2)
        self.assertEqual(max(c for _, c in axes) + 1, 3)
        xs, ys = np.unique(coords[:, 0]), np.unique(coords[:, 1])
        for k, (x, y) in enumerate(coords):
            row = len(ys) - 1 - int(np.searchsorted(ys, y))
            col = int(np.searchsorted(xs, x))
            self.assertAlmostEqual(self._value_at(axes, row, col), k + 1, places=6)
        # explicit corners: top row = largest y, left column = smallest x
        self.assertAlmostEqual(self._value_at(axes, 0, 0), 2, places=6)   # (-10, 5)
        self.assertAlmostEqual(self._value_at(axes, 1, 2), 1, places=6)   # (10, -5)

    def test_origin_lower_and_bright_pixel_at_bottom(self):
        img = np.zeros((6, 6))
        img[0, 3] = 1.0
        tiledDisplay([SimpleNamespace(sampling=img)])
        im = plt.gcf().axes[0].images[0]
        self.assertEqual(im.origin, 'lower')
        left, right, bottom, top = im.get_extent()
        self.assertAlmostEqual(bottom, -0.5)
        self.assertAlmostEqual(top, 5.5)
        arr = im.get_array()
        self.assertEqual(np.unravel_index(np.argmax(arr), arr.shape), (0, 3))

    def test_origin_all_images(self):
        tiledDisplay([_psf(k + 1) for k in range(4)])
        for ax in plt.gcf().axes:
            self.assertEqual(ax.images[0].origin, 'lower')

    def _check_index_layout(self, fig):
        axes = _grid_by_position(fig)
        self.assertEqual(len(fig.axes), 4)
        self.assertEqual(max(r for r, _ in axes) + 1, 2)
        self.assertEqual(max(c for _, c in axes) + 1, 2)
        # PSF k at row nrows-1-k//ncols, col k%ncols
        for k in range(4):
            self.assertAlmostEqual(self._value_at(axes, 1 - k // 2, k % 2), k + 1, places=6)
        self.assertAlmostEqual(self._value_at(axes, 1, 0), 1, places=6)  # PSF 0 bottom left
        self.assertAlmostEqual(self._value_at(axes, 0, 1), 4, places=6)  # PSF 3 top right

    def test_fallback_no_coords(self):
        tiledDisplay([_psf(k + 1) for k in range(4)], coords=None)
        self._check_index_layout(plt.gcf())

    def test_fallback_coords_not_full_grid(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(-10, 10, size=(4, 2))
        tiledDisplay([_psf(k + 1) for k in range(4)], coords=coords)
        self._check_index_layout(plt.gcf())

    def test_single_psf(self):
        tiledDisplay([_psf(3)])
        fig = plt.gcf()
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig.axes[0].images), 1)
        # also with coords given
        plt.close('all')
        tiledDisplay([_psf(3)], coords=np.array([[1.0, 2.0]]))
        self.assertEqual(len(plt.gcf().axes), 1)

    def test_standard_psf_plot_origin(self):
        img = np.zeros((8, 8))
        img[0, 4] = 1.0
        standardPsfPlot(img)
        fig = plt.gcf()
        im = fig.axes[0].images[0]
        self.assertEqual(im.origin, 'lower')
        self.assertAlmostEqual(im.get_extent()[2], -0.5)


if __name__ == '__main__':
    unittest.main()
