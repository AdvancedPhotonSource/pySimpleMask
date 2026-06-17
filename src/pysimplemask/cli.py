"""Console scripts for pysimplemask."""

import argparse
import logging
import os
import sys

from pysimplemask import __version__
from pysimplemask.core.partition import combine_qmap_files


def main():
    """Launch the pySimpleMask GUI."""
    parser = argparse.ArgumentParser(
        "pySimpleMask: A GUI for creating mask and q-partition maps "
        "for scattering patterns in preparation for SAXS/WAXS/XPCS data reduction"
    )
    parser.add_argument("--path", "-p", required=False, default=os.getcwd())
    parser.add_argument(
        "--version", action="version", version=f"pySimpleMask {__version__}"
    )
    args = parser.parse_args()
    from pysimplemask.gui.app import main_gui

    sys.exit(main_gui(args.path))


def combine_qmaps():
    """CLI entry point: combine two qmap HDF5 files into one."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-combine-qmaps",
        description="Combine two pySimpleMask qmap HDF5 files into a single output file.",
    )
    parser.add_argument("qmap_file1", help="Path to the first qmap HDF5 file.")
    parser.add_argument("qmap_file2", help="Path to the second qmap HDF5 file.")
    parser.add_argument("output_file", help="Path for the combined output qmap HDF5 file.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    combine_qmap_files(args.qmap_file1, args.qmap_file2, args.output_file)


def _build_qmap_args(argv=None):
    """Parse arguments for build-qmap; returns a Namespace. Testable without sys.argv."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-build-qmap",
        description=(
            "Build a q-partition map (qmap) from a raw scattering file. "
            "Replicates the GUI workflow: load → center → mask → partition → save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── positional ──────────────────────────────────────────────────────────
    parser.add_argument(
        "dataset",
        help="Path to the raw scattering file (.hdf, .h5, .imm, .bin, …)",
    )

    # ── data loading ────────────────────────────────────────────────────────
    grp_load = parser.add_argument_group("data loading")
    grp_load.add_argument(
        "--beamline",
        default="APS_8IDI",
        choices=["APS_8IDI", "APS_9IDD"],
        help="Beamline reader.",
    )
    grp_load.add_argument(
        "--begin-idx",
        type=int,
        default=0,
        metavar="N",
        help="First frame index to include.",
    )
    grp_load.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        metavar="N",
        help="Frames to average. 0=all, -1=representative subset.",
    )

    # ── beam center ─────────────────────────────────────────────────────────
    grp_cen = parser.add_argument_group("beam center")
    grp_cen.add_argument(
        "--no-find-center",
        action="store_true",
        help="Skip goto_max + find_center; use metadata center as-is.",
    )
    grp_cen.add_argument(
        "--max-radius",
        type=int,
        default=384,
        metavar="N",
        help="Crop half-size (px) passed to find_center for speed on large detectors.",
    )

    # ── mask ────────────────────────────────────────────────────────────────
    grp_mask = parser.add_argument_group("mask")
    grp_mask.add_argument(
        "--blemish",
        default=None,
        metavar="FILE",
        help="Blemish/bad-pixel file (.tif or .h5).",
    )
    grp_mask.add_argument(
        "--blemish-key",
        default="/qmap/mask",
        metavar="KEY",
        help="HDF5 dataset path inside the blemish file.",
    )
    grp_mask.add_argument(
        "--threshold-high",
        type=float,
        default=None,
        metavar="VAL",
        help="Mask pixels with intensity >= VAL (raw counts).",
    )

    # ── partition ───────────────────────────────────────────────────────────
    grp_part = parser.add_argument_group("partition")
    grp_part.add_argument(
        "--mode",
        default="q-phi",
        choices=["q-phi", "x-y", "eq-ephi"],
        help="Partition axes.",
    )
    grp_part.add_argument("--dq-num", type=int, default=10, metavar="N",
                          help="Dynamic q bins.")
    grp_part.add_argument("--sq-num", type=int, default=100, metavar="N",
                          help="Static q bins (must be a multiple of --dq-num).")
    grp_part.add_argument("--dp-num", type=int, default=36, metavar="N",
                          help="Dynamic phi bins.")
    grp_part.add_argument("--sp-num", type=int, default=360, metavar="N",
                          help="Static phi bins (must be a multiple of --dp-num).")
    grp_part.add_argument("--phi-offset", type=float, default=0.0, metavar="DEG",
                          help="Phi axis offset in degrees.")
    grp_part.add_argument("--symmetry-fold", type=int, default=1, metavar="N",
                          help="Rotational symmetry fold.")
    grp_part.add_argument(
        "--style",
        default="linear",
        choices=["linear", "logarithmic"],
        help="Bin spacing style.",
    )

    # ── output ──────────────────────────────────────────────────────────────
    grp_out = parser.add_argument_group("output")
    grp_out.add_argument(
        "--output-qmap",
        default="qmap.hdf",
        metavar="FILE",
        help="Output qmap HDF5 path.",
    )
    grp_out.add_argument(
        "--output-mask",
        default="mask.tif",
        metavar="FILE",
        help="Output mask TIFF path. Pass empty string to skip.",
    )

    # ── logging ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )

    return parser.parse_args(argv)


def _run_build_qmap(args):
    """Execute the qmap-build pipeline. Separated from argument parsing for testability."""
    from pysimplemask.core import SimpleMaskModel

    m = SimpleMaskModel()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    ok = m.read_data(
        args.dataset,
        beamline=args.beamline,
        begin_idx=args.begin_idx,
        num_frames=args.num_frames,
    )
    if not ok:
        raise RuntimeError(f"Failed to load dataset: {args.dataset}")
    logging.info("Loaded %s  shape=%s", args.dataset, m.shape)

    # ── 2. Beam center ───────────────────────────────────────────────────────
    if not args.no_find_center:
        center_vh = m.goto_max()
        logging.info(
            "goto_max center: row=%.1f col=%.1f", center_vh[0], center_vh[1]
        )
        refined_vh = m.find_center(max_radius=args.max_radius)
        logging.info(
            "find_center result: row=%.1f col=%.1f", refined_vh[0], refined_vh[1]
        )
        m.dset.set_center_vh(refined_vh)
        m.update_parameters()

    # ── 3. Mask ──────────────────────────────────────────────────────────────
    if args.blemish:
        m.mask_evaluate("mask_blemish", fname=args.blemish, key=args.blemish_key)
        m.mask_apply("mask_blemish")
        logging.info("Applied blemish: %s", args.blemish)

    if args.threshold_high is not None:
        m.mask_evaluate(
            "mask_threshold",
            low=0,
            high=args.threshold_high,
            low_enable=False,
            high_enable=True,
        )
        m.mask_apply("mask_threshold")
        logging.info("Applied threshold-high: %s", args.threshold_high)

    bad = int(m.mask.size - m.mask.sum())
    logging.info(
        "Final mask: %d pixels masked (%.2f%%)",
        bad,
        bad / m.mask.size * 100,
    )

    # ── 4. Partition ─────────────────────────────────────────────────────────
    m.compute_partition(
        mode=args.mode,
        dq_num=args.dq_num,
        sq_num=args.sq_num,
        dp_num=args.dp_num,
        sp_num=args.sp_num,
        phi_offset=args.phi_offset,
        symmetry_fold=args.symmetry_fold,
        style=args.style,
    )
    logging.info("Partition computed (mode=%s)", args.mode)

    # ── 5. Save ──────────────────────────────────────────────────────────────
    m.save_partition(args.output_qmap)
    logging.info("Saved qmap: %s", args.output_qmap)

    if args.output_mask:
        m.save_mask(args.output_mask)
        logging.info("Saved mask: %s", args.output_mask)


def build_qmap():
    """CLI entry point: build a qmap from a raw scattering file."""
    args = _build_qmap_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _run_build_qmap(args)
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
