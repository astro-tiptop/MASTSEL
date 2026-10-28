# Perfect coronagraph: end-to-end validation with SPECULA

This is a validation, not part of the test suite. It checks the MASTSEL perfect-coronagraph image (`psdSetToPsfSet(..., perfectCoronagraph=True)`) against a frame-by-frame perfect coronagraph applied to the residual fields of a SPECULA end-to-end simulation.

## Files

| File | Content |
|---|---|
| `params_eris_ngs.yml` | ERIS NGS SCAO: 8 m pupil (160 px, obscuration 0.138), SH 40×40, 702 modes. |
| `calib_eris_ngs_subaps.yml`, `calib_eris_ngs_rec.yml` | Calibration of the subapertures and of the interaction/reconstruction matrices (`eris_subaps`, `eris_rec_test`). |
| `coro_override.yml` | The overrides for this run: on-axis NGS, four `PsfCoronagraph` objects (J, H, K, L at nd = 4), 4 s, and the residual field saved every 4 ms. |
| `specula_paired_comparison.py` | The comparison, run on the saved fields. |
| `specula_eris_reference.npz` | Radial profiles (contrast units) of the frame-by-frame image and of the long-exposure formulas, and 128×128 crops of the long-exposure PSF and of the frame-by-frame coronagraphic image, for t = 0.1–4.0 s. |

## How to run

The calibration data go in the `root_dir` of `params_eris_ngs.yml`. The DM influence functions (`VLT_ifunc`) are not distributed here.

```
specula params_eris_ngs.yml calib_eris_ngs_subaps.yml   # subapertures
specula params_eris_ngs.yml calib_eris_ngs_rec.yml      # interaction and reconstruction matrices
specula params_eris_ngs.yml coro_override.yml           # ~45 s on a Quadro RTX 6000; ~0.2 GB of fields in ./out_ef
python specula_paired_comparison.py out_ef/<run>/ef.fits 4.0 specula_eris_reference.npz
```

## The run

- The loop is stable over the 4 s. In 0.5 s windows, the residual is 102–104 nm RMS, the tip-tilt is 16–24 nm RMS, and the K-band |a|² is 0.915–0.919.
- The long-exposure SR is J 0.77, H 0.86, K 0.92, L 0.97.
- The script computes the frame-by-frame image from 975 fields (every 4 ms). It agrees with SPECULA `PsfCoronagraph`, which integrates every 1 ms, within 0.8 %: the ratio is 0.992–1.004 per annulus in all bands.

## Result

The table gives the ratio of each long-exposure formula to the frame-by-frame image, per annulus. The range covers the four bands:

| r [λ/D] | coherent removal, ⟨I⟩ − \|⟨a⟩\|² PSF_DL | SR removal, ⟨I⟩ − SR PSF_DL | Gaussian formula (MASTSEL), empirical g |
|---|---|---|---|
| 0–0.5 | 0.009–1.34 | 0.000–1.27 | 1.17–1.41 |
| 0.5–1 | 1.26–1.30 | 1.26–1.29 | 1.04–1.07 |
| 1–1.5 | 0.90–0.96 | 0.90–0.96 | 0.95–1.02 |
| 1.5–2 | 0.89–0.99 | 0.89–0.99 | 1.00–1.05 |
| 2–5 | 0.94–0.97 | 0.94–0.97 | 0.96–0.99 |
| 5–40 | 0.97–1.00 | 0.97–1.00 | 0.98–1.00 |

### Why the simpler formulas fail near the axis

The frame-by-frame image is

⟨|FT(E − a P)|²⟩ = ⟨I⟩ − 2 Re[⟨a* FT(E)⟩ conj(FT P)] + ⟨|a|²⟩ |FT P|².

Both simpler formulas assume ⟨a* FT(E)⟩ = ⟨|a|²⟩ FT(P), i.e. that the part of the field correlated with a has the shape of the diffraction-limited field. This is not true: the residual phase correlates a with the scattered field within 1–2 λ/D.

- The coherent removal also replaces ⟨|a|²⟩ = SR by |⟨a⟩|². The difference is the frame-to-frame fluctuation of the coherent energy, dominated by tip-tilt jitter. It is small in this run (|⟨a⟩|² = SR to 3 digits), but it was large in an earlier run with ~200–500 nm of tip-tilt.
- In the L band, within 0.5 λ/D, both remove almost everything (ratio 0.01 and 0.00). Between 0.5 and 1 λ/D they overestimate by 26–30 % in every band.

### The MASTSEL formula

- The script evaluates it with the empirical g = OTF_LE / OTF_tel of the same frames. This tests the Gaussian/stationary assumption of the formula, not the MASTSEL code; the code is covered by the Monte Carlo test.
- It computes ⟨a* FT(E)⟩ from the second-order moments of the field and is exact for a Gaussian, stationary phase.
- It is within 7 % of the frame-by-frame image beyond 0.5 λ/D and within 5 % beyond 2 λ/D.
- Within 0.5 λ/D it overestimates by 17–41 %. This is plausibly due to the non-Gaussian, non-stationary part of the real residual, but that is a hypothesis, not demonstrated.
