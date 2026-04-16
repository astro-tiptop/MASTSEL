#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from mastsel.mavisPsf import psdSetToPsfSet

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "psd")


def _load_fixture_paths():
    if not os.path.isdir(FIXTURE_DIR):
        return []
    return [
        os.path.join(FIXTURE_DIR, name)
        for name in sorted(os.listdir(FIXTURE_DIR))
        if name.endswith(".npz")
    ]


def _flatten_psf_set(psf_set):
    if isinstance(psf_set[0], list):
        return [item.sampling for row in psf_set for item in row]
    return [item.sampling for item in psf_set]


def _normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    s = arr.sum()
    if s == 0:
        return arr
    return arr / s


def _peak_rel_diff(psf_a, psf_b):
    a = _normalize(psf_a)
    b = _normalize(psf_b)
    return abs(b.max() - a.max()) / max(a.max(), 1e-20)


def _l2_rel_diff(psf_a, psf_b):
    a = _normalize(psf_a)
    b = _normalize(psf_b)
    denom = np.linalg.norm(a)
    if denom <= 0:
        return 0.0
    return np.linalg.norm(b - a) / denom


def _zero_nyquist_row_col(psd):
    out = np.asarray(psd).copy()
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError("PSD must be a square 2D array")
    if out.shape[0] % 2 == 0:
        out[0, :] = 0.0
        out[:, 0] = 0.0
    return out


def _pad_or_crop_centered(arr, target_n):
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n == target_n:
        return arr
    if target_n < n:
        start = n // 2 - target_n // 2
        return arr[start : start + target_n, start : start + target_n]

    out = np.zeros((target_n, target_n), dtype=arr.dtype)
    start = target_n // 2 - n // 2
    out[start : start + n, start : start + n] = arr
    return out


def _expand_with_nyquist_row_col(psd):
    arr = np.asarray(psd)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("PSD must be a square 2D array")
    n = arr.shape[0]
    if n % 2 != 0:
        return arr.copy()

    out = np.zeros((n + 1, n + 1), dtype=arr.dtype)
    out[1:, 1:] = arr
    out[0, 1:] = arr[0, :]
    out[1:, 0] = arr[:, 0]
    out[0, 0] = arr[0, 0]
    return out


def _run_reference(data):
    N = int(data["N"])
    input_psds = np.asarray(data["input_psds"])

    kwargs = dict(
        mask=np.asarray(data["mask"]),
        wavelength=np.asarray(data["wavelength"]),
        N=N,
        nPixPup=int(data["nPixPup"]),
        grid_diameter=float(data["grid_diameter"]),
        freq_range=float(data["freq_range"]),
        dk=float(data["dk"]),
        nPixPsf=int(data["nPixPsf"]),
        wvlRef=float(data["wvlRef"]),
        oversampling=float(data["oversampling"]),
        padPSD=bool(data["padPSD"]),
    )
    has_opd = bool(data["has_opd"])
    if has_opd:
        kwargs["opdMap"] = np.asarray(data["opd_map"])

    base = psdSetToPsfSet(input_psds, **kwargs)
    return _flatten_psf_set(base)


def _run_zeroed_case(data):
    N = int(data["N"])
    input_psds = np.asarray(data["input_psds"])
    zeroed_psds = np.asarray([_zero_nyquist_row_col(psd) for psd in input_psds])

    kwargs = dict(
        mask=np.asarray(data["mask"]),
        wavelength=np.asarray(data["wavelength"]),
        N=N,
        nPixPup=int(data["nPixPup"]),
        grid_diameter=float(data["grid_diameter"]),
        freq_range=float(data["freq_range"]),
        dk=float(data["dk"]),
        nPixPsf=int(data["nPixPsf"]),
        wvlRef=float(data["wvlRef"]),
        oversampling=float(data["oversampling"]),
        padPSD=bool(data["padPSD"]),
    )
    has_opd = bool(data["has_opd"])
    if has_opd:
        kwargs["opdMap"] = np.asarray(data["opd_map"])

    zeroed = psdSetToPsfSet(zeroed_psds, **kwargs)
    return _flatten_psf_set(zeroed)


def _run_expanded_case(data):
    N = int(data["N"])
    input_psds = np.asarray(data["input_psds"])
    expanded_psds = np.asarray([_expand_with_nyquist_row_col(psd) for psd in input_psds])

    odd_N = N + 1
    odd_nPixPsf = int(data["nPixPsf"]) + 1
    odd_nPixPup = int(data["nPixPup"])
    odd_grid = float(data["grid_diameter"])
    psd_step = float(data["freq_range"]) / float(N)
    odd_freq = odd_N * psd_step

    kwargs = dict(
        mask=np.asarray(data["mask"]),
        wavelength=np.asarray(data["wavelength"]),
        N=odd_N,
        nPixPup=odd_nPixPup,
        grid_diameter=odd_grid,
        freq_range=odd_freq,
        dk=float(data["dk"]),
        nPixPsf=odd_nPixPsf,
        wvlRef=float(data["wvlRef"]),
        oversampling=float(data["oversampling"]),
        padPSD=bool(data["padPSD"]),
    )
    has_opd = bool(data["has_opd"])
    if has_opd:
        kwargs["opdMap"] = np.asarray(data["opd_map"])

    expanded = psdSetToPsfSet(expanded_psds, **kwargs)
    return _flatten_psf_set(expanded)


def _diff_rows(base_psfs, other_psfs, mode_name):
    rows = []
    for idx, (psf_base, psf_other) in enumerate(zip(base_psfs, other_psfs)):
        base_aligned = np.asarray(psf_base)
        other_aligned = _pad_or_crop_centered(psf_other, base_aligned.shape[0])
        base_n = _normalize(base_aligned)
        other_n = _normalize(other_aligned)
        rows.append(
            {
                "mode": mode_name,
                "psf_idx": idx,
                "base_peak": float(base_n.max()),
                "other_peak": float(other_n.max()),
                "peak_rel_diff": _peak_rel_diff(base_aligned, other_aligned),
                "l2_rel_diff": _l2_rel_diff(base_aligned, other_aligned),
                "base_psf": base_n,
                "other_psf": other_n,
            }
        )
    return rows


def _print_table(rows):
    headers = [
        "case",
        "N",
        "mode",
        "psf_idx",
        "base_peak",
        "other_peak",
        "peak_rel_diff_%",
        "l2_rel_diff_%",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["case"],
                str(row["N"]),
                row["mode"],
                str(row["psf_idx"]),
                f"{row['base_peak']:.6e}",
                f"{row['other_peak']:.6e}",
                f"{100.0 * row['peak_rel_diff']:.6f}",
                f"{100.0 * row['l2_rel_diff']:.6f}",
            ]
        )

    widths = [len(h) for h in headers]
    for trow in table_rows:
        for i, value in enumerate(trow):
            widths[i] = max(widths[i], len(value))

    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(widths)))
    print(line)
    print(sep)
    for trow in table_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(trow)))


def _print_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["mode"], row["case"])
        grouped.setdefault(key, []).append(row)

    print("\nSummary by case/mode")
    print("mode | case | n_psf | mean_peak_rel_% | max_peak_rel_% | mean_l2_rel_% | max_l2_rel_%")
    for (mode, case), items in sorted(grouped.items()):
        peak_vals = np.asarray([100.0 * r["peak_rel_diff"] for r in items], dtype=np.float64)
        l2_vals = np.asarray([100.0 * r["l2_rel_diff"] for r in items], dtype=np.float64)
        print(
            f"{mode} | {case} | {len(items)} | "
            f"{peak_vals.mean():.6f} | {peak_vals.max():.6f} | "
            f"{l2_vals.mean():.6f} | {l2_vals.max():.6f}"
        )


def _safe_filename(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _plot_psf_triptych(row, output_dir):
    import matplotlib.pyplot as plt

    base = row["base_psf"]
    other = row["other_psf"]
    diff = other - base
    vmax = max(float(np.max(np.abs(diff))), 1e-20)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axs[0].imshow(base, origin="lower")
    axs[0].set_title("baseline")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(other, origin="lower")
    axs[1].set_title(row["mode"])
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axs[2].set_title("difference")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    title = (
        f"{row['case']} N={row['N']} psf#{row['psf_idx']} | "
        f"peak diff={100.0 * row['peak_rel_diff']:.6f}%"
    )
    fig.suptitle(title)

    out_name = (
        f"psf_{_safe_filename(row['case'])}_N{row['N']}_{row['mode']}"
        f"_idx{row['psf_idx']}.png"
    )
    fig.savefig(os.path.join(output_dir, out_name), dpi=150)
    plt.close(fig)


def _plot_peak_profiles(rows, output_dir):
    import matplotlib.pyplot as plt

    grouped = {}
    for row in rows:
        key = (row["case"], row["N"], row["mode"])
        grouped.setdefault(key, []).append(row)

    for (case, n_val, mode), items in sorted(grouped.items()):
        items = sorted(items, key=lambda x: x["psf_idx"])
        x = np.asarray([it["psf_idx"] for it in items], dtype=int)
        base_peak = np.asarray([it["base_peak"] for it in items], dtype=np.float64)
        other_peak = np.asarray([it["other_peak"] for it in items], dtype=np.float64)
        rel = np.asarray([100.0 * it["peak_rel_diff"] for it in items], dtype=np.float64)

        fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax1.plot(x, base_peak, marker="o", label="baseline peak")
        ax1.plot(x, other_peak, marker="s", label=f"{mode} peak")
        ax1.set_xlabel("psf index")
        ax1.set_ylabel("normalized peak")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, rel, color="tab:red", marker="^", linestyle="--", label="rel diff %")
        ax2.set_ylabel("peak relative diff [%]")

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
        ax1.set_title(f"{case} N={n_val} mode={mode}")

        out_name = f"peaks_{_safe_filename(case)}_N{n_val}_{mode}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Report impact of zeroing Nyquist row/col in fixture PSDs")
    parser.add_argument("--all", action="store_true", help="Include odd-N fixtures too (no zeroing applied there)")
    parser.add_argument("--format", choices=["table", "csv"], default="table", help="Output format")
    parser.add_argument("--plot-psf", action="store_true", help="Save baseline/other/difference PSF images")
    parser.add_argument("--plot-peaks", action="store_true", help="Save 1D peak summary plots")
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "report_plots"), help="Directory for optional plots")
    parser.add_argument("--max-psf-plots", type=int, default=20, help="Maximum number of PSF triptych plots to generate")
    args = parser.parse_args()

    fixture_paths = _load_fixture_paths()
    if not fixture_paths:
        raise SystemExit("No fixture files found in tests/fixtures/psd")

    rows_all = []
    for fixture_path in fixture_paths:
        data = np.load(fixture_path, allow_pickle=False)
        case_name = str(data["case_name"])
        N = int(data["N"])
        if not args.all and N % 2 != 0:
            continue

        base_psfs = _run_reference(data)
        rows = _diff_rows(base_psfs, _run_zeroed_case(data), "zero_row_col")
        rows.extend(_diff_rows(base_psfs, _run_expanded_case(data), "expand_to_n_plus_1"))
        for row in rows:
            row["case"] = case_name
            row["N"] = N
            rows_all.append(row)

    if args.format == "csv":
        print("case,N,mode,psf_idx,base_peak,other_peak,peak_rel_diff_percent,l2_rel_diff_percent")
        for row in rows_all:
            print(
                f"{row['case']},{row['N']},{row['mode']},{row['psf_idx']},"
                f"{row['base_peak']:.12e},{row['other_peak']:.12e},"
                f"{100.0 * row['peak_rel_diff']:.6f},{100.0 * row['l2_rel_diff']:.6f}"
            )
    else:
        _print_table(rows_all)
        _print_summary(rows_all)

    if args.plot_psf or args.plot_peaks:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_psf:
        count = 0
        for row in rows_all:
            _plot_psf_triptych(row, args.output_dir)
            count += 1
            if count >= args.max_psf_plots:
                break
        print(f"\nSaved {count} PSF triptych plots to: {args.output_dir}")

    if args.plot_peaks:
        _plot_peak_profiles(rows_all, args.output_dir)
        print(f"Saved peak summary plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
