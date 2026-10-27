# Perfect coronagraph: end-to-end validation with SPECULA

This is a validation, not part of the test suite. It checks how the MASTSEL perfect-coronagraph image (`psdSetToPsfSet(..., perfectCoronagraph=True)`) compares with a frame-by-frame perfect coronagraph applied to the residual fields of a SPECULA end-to-end simulation.

## Files

| File | Content |
|---|---|
| `params_eris_ngs.yml` | The ERIS NGS SCAO configuration: SH 40×40, 702 modes, VLT pupil. |
| `coro_override.yml` | The overrides for this run: on-axis NGS, four `PsfCoronagraph` objects (J, H, K, L at nd = 4), 4 s, and the residual field saved every 4 ms. |
| `specula_paired_comparison.py` | The comparison, run on the saved fields. |
| `specula_eris_reference.npz` | Radial profiles (contrast units) and 128×128 crops of the long-exposure PSF and of the frame-by-frame coronagraphic image, for t = 0.1–2.0 s. |

## How to run

```
specula params_eris_ngs.yml coro_override.yml        # ~75 s on a Quadro RTX 6000; ~1 GB of fields in ./out_ef
python specula_paired_comparison.py out_ef/<run>/ef.fits 2.0 specula_eris_reference.npz
```

The calibration data are in the `root_dir` of `params_eris_ngs.yml`.

## Result

The frame-by-frame computation from the saved fields agrees with SPECULA `PsfCoronagraph` to 0.1 % (ratio 0.999–1.002 per annulus). This was checked on the full 4 s run and is not part of the script.

The table gives the ratio of each long-exposure formula to the frame-by-frame image, per annulus. The range covers the four bands, J to L, with SR between 0.32 and 0.86:

| r [λ/D] | long-exposure coherent removal | SR·PSF_DL removal | Gaussian frame-by-frame formula, empirical g |
|---|---|---|---|
| 0–0.5 | 0.40–4.1 | 0.02–1.40 | 0.77–0.87 |
| 0.5–1 | 0.81–1.29 | 0.80–1.05 | 0.88–0.94 |
| 1.5–2 (first dark ring) | 1.13–1.43 | 1.04–1.40 | 1.09–1.62 |
| 2–5 | 0.95–1.09 | 0.94–1.05 | 0.98–1.20 |
| > 5 | 0.93–1.01 | 0.93–0.99 | 0.97–1.05 |

- **Long-exposure coherent removal.** Removing e^{−σ²}·PSF_DL misses ⟨|a|²⟩ − |⟨a⟩|², the temporal fluctuation of the coherent energy, which is dominated by tip-tilt and the other low-order modes. This is why the near-axis ratios in the first column are far from 1.
- **The last column.** It is the formula used by MASTSEL, evaluated by the script with the empirical g = OTF_LE/OTF_tel of the same frames, so it tests the Gaussian/stationary assumption, not the MASTSEL code (that is covered by the Monte Carlo test). The formula is exact for a Gaussian, stationary phase.
- **What remains.** The deviation is plausibly due to the non-Gaussian, non-stationary part of the real residual (pupil edge, spiders, finite statistics), but this is a hypothesis, not demonstrated. It is largest in the first dark ring at high SR.

## Known problem of this run

After about 2.2 s the simulation degrades:
- the instantaneous K-band SR falls from ~0.8 to 0.03–0.3;
- the tip-tilt grows from ~190 nm to ~500 nm RMS.

The same happens with integrator gain 0.2, so it is not a loop instability. It is not investigated here; possible causes are the phase screens or the calibration. For this reason only t < 2 s is used.

For the same reason, a direct TIPTOP vs SPECULA comparison is not meaningful with this configuration: TIPTOP gives a K-band SR of 0.94, SPECULA 0.65 on t < 2 s.
