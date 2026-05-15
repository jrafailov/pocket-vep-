"""Render the ABL1 A433T case study figure for the report.

Pulls the AlphaFold structure for ABL1 (UniProt P00519), highlights residue
A433 (the variant the all-features model rescued from the sequence-only
model's missed call), and renders the surrounding pocket as a semi-
transparent surface. Pocket residues are looked up from structure_features
(in_pocket flag) and intersected with a 8 Angstrom spatial neighborhood
of A433 so we only show the pocket the variant actually sits in.

Output
    report/figures/case_study.png  (and .pdf via pymol's raytrace + convert)

Usage
    python scripts/visualize_case_study.py
    python scripts/visualize_case_study.py --uniprot P00519 --position 433 --gene ABL1
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
from pymol import cmd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDB_DIR = ROOT / "data/raw/alphafold"
DEFAULT_STRUCT_FEATURES = ROOT / "data/processed/structure_features.parquet"
DEFAULT_OUT = ROOT / "report/figures/case_study.png"


def _decompress_pdb(src_gz: Path) -> Path:
    """PyMOL handles .pdb directly but the cache stores .pdb.gz. Decompress
    to a tempfile and return the path."""
    tmp = Path(tempfile.mkdtemp(prefix="case_study_")) / src_gz.name.replace(".gz", "")
    with gzip.open(src_gz, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return tmp


def _load_panel_label_font(size: int):
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _overlay_panel_labels(png_path: Path, panel_labels: tuple[str, str],
                          top_strip_frac: float = 0.06) -> None:
    """Replace the small matplotlib panel titles with PIL-drawn ones that
    read well at the report's figure scale. matplotlib title fonts scale
    unpredictably under bbox_inches='tight', so we white out the top strip
    and redraw at a fixed fraction of image height."""
    from PIL import Image, ImageDraw

    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    strip_h = int(h * top_strip_frac)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, strip_h)], fill=(255, 255, 255, 255))

    font_size = max(48, int(h * 0.045))
    font = _load_panel_label_font(font_size)

    left_x = int(w * 0.015)
    right_x = int(w * 0.515)
    y = int(strip_h * 0.05)
    draw.text((left_x, y), panel_labels[0], fill=(0, 0, 0, 255), font=font)
    draw.text((right_x, y), panel_labels[1], fill=(0, 0, 0, 255), font=font)
    img.save(png_path)


def _composite(wide_png: Path, close_png: Path, out_png: Path,
               panel_labels: tuple[str, str]) -> None:
    """Stitch two PyMOL-rendered PNGs into one side-by-side figure. We render
    without matplotlib titles and overlay the panel labels with PIL at a
    print-readable size."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from _plot_style import apply_style

    apply_style()
    wide = mpimg.imread(wide_png)
    close = mpimg.imread(close_png)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, img in zip(axes, [wide, close]):
        ax.imshow(img)
        ax.set_axis_off()
    fig.tight_layout(pad=0.4)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    _overlay_panel_labels(out_png, panel_labels)


def _pocket_residues_near(uniprot_id: str, position: int,
                          struct_features: Path) -> list[int]:
    """Return positions of in_pocket residues for this protein. The 3D
    intersection with a spatial neighborhood of `position` happens later
    inside PyMOL, since we need 3D coordinates for that."""
    df = pd.read_parquet(struct_features)
    rows = df[(df.uniprot_id == uniprot_id) & (df.in_pocket == 1)]
    return sorted(rows.position.astype(int).tolist())


def _setup_scene(pdb: Path, position: int, in_pocket_positions: list[int],
                 neighborhood_radius: float) -> str:
    """Load the protein, build all the visual elements (cartoon, variant
    sticks/spheres, pocket mesh, pocket-lining residues), and return the
    PyMOL selection name for the variant residue. The camera is NOT set
    here; that's per-view."""
    cmd.feedback("disable", "all", "everything")
    cmd.reinitialize()
    cmd.load(str(pdb), "prot")
    cmd.remove("solvent")
    cmd.hide("everything")

    cmd.show("cartoon", "prot")
    cmd.color("gray70", "prot")
    cmd.set("cartoon_transparency", 0.0)
    # Hide disordered / low-confidence regions. AlphaFold writes its
    # per-residue pLDDT into the PDB B-factor column, so `b < 50` selects
    # the residues AlphaFold flags as poorly modeled. These appear as
    # extended spaghetti loops in the wide view if shown.
    cmd.hide("cartoon", "prot and b < 50")

    sel_var = f"resi {position} and prot"
    cmd.show("sticks", sel_var)
    cmd.show("spheres", f"{sel_var} and name CA")
    cmd.color("firebrick", sel_var)
    cmd.set("stick_radius", 0.35, sel_var)
    cmd.set("sphere_scale", 0.55, sel_var)

    pos_list_pml = "+".join(str(p) for p in in_pocket_positions)
    cmd.select("all_pockets", f"resi {pos_list_pml} and prot")
    cmd.select("local_pocket",
               f"byres (all_pockets within {neighborhood_radius} of "
               f"({sel_var} and name CA))")

    cmd.show("mesh", "local_pocket")
    cmd.color("skyblue", "local_pocket")
    cmd.set("mesh_width", 0.6, "local_pocket")
    cmd.set("transparency", 0.30, "local_pocket")

    cmd.color("marine", f"local_pocket and not ({sel_var})")
    cmd.color("firebrick", sel_var)

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.35)
    cmd.set("specular", 0.4)

    return sel_var


def _render_view(sel_var: str, view: str, out_png: Path,
                 width: int = 1400, height: int = 1300) -> None:
    """Render one of two camera views to a PNG. `view='wide'` shows the
    whole protein with the variant residue still visible; `view='close'`
    zooms onto the pocket."""
    cmd.orient("prot and b > 50")
    if view == "wide":
        # Whole-protein context. AlphaFold disordered tails (b < 50) are
        # hidden so we zoom on just the high-confidence cartoon to avoid
        # leaving the rendered structure tiny in a sea of empty whitespace.
        cmd.center("prot and b > 50")
        cmd.zoom("prot and b > 50", buffer=3.0)
    elif view == "close":
        # Zoom on the variant + pocket. Same protein-axes orientation as
        # the wide view so the reader can connect the two panels visually.
        cmd.center(sel_var)
        cmd.zoom("local_pocket or (resi {} and prot)".format(
            sel_var.split()[1]), buffer=6.0)
    else:
        raise ValueError(f"unknown view: {view!r}")
    cmd.turn("y", 30)
    cmd.turn("x", -10)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(out_png), width=width, height=height, dpi=300, ray=1)


def render(uniprot_id: str, position: int, gene: str, variant_label: str,
           pdb_dir: Path, struct_features: Path, out_png: Path,
           neighborhood_radius: float = 8.0) -> None:
    pdb_gz = pdb_dir / f"AF-{uniprot_id}-F1-model_v4.pdb.gz"
    if not pdb_gz.exists():
        raise FileNotFoundError(f"AlphaFold PDB not found for {uniprot_id}: {pdb_gz}")
    pdb = _decompress_pdb(pdb_gz)

    in_pocket_positions = _pocket_residues_near(uniprot_id, position, struct_features)
    if not in_pocket_positions:
        raise RuntimeError(f"No in_pocket residues found for {uniprot_id}")
    if position not in in_pocket_positions:
        print(f"[case-study] note: target residue {position} is NOT itself "
              f"in a pocket per structure_features; rendering the spatially-"
              f"nearest pocket instead.")

    sel_var = _setup_scene(pdb, position, in_pocket_positions, neighborhood_radius)

    n_pocket = cmd.count_atoms("local_pocket and name CA")
    print(f"[case-study] {n_pocket} pocket residues within "
          f"{neighborhood_radius} A of {gene} residue {position}")
    if n_pocket == 0:
        raise RuntimeError(
            f"No pocket residues found within {neighborhood_radius} A of "
            f"residue {position}; either the residue is not near any pocket "
            f"or the radius is too small.")

    # Render the two views to temp files, then composite side-by-side.
    tmp_dir = Path(tempfile.mkdtemp(prefix="case_study_panels_"))
    wide_png = tmp_dir / "wide.png"
    close_png = tmp_dir / "close.png"
    _render_view(sel_var, "wide", wide_png)
    _render_view(sel_var, "close", close_png)

    _composite(wide_png, close_png, out_png,
               panel_labels=("A   Full protein", "B   Pocket close-up"))
    print(f"[case-study] wrote {out_png.relative_to(ROOT)}")

    # Also dump a PDF via pymol's vector-ish ray. PyMOL's pdf export is
    # actually a wrapper around the rasterized PNG, so we just inform the
    # user; the .png is the real figure.
    out_pdf = out_png.with_suffix(".pdf")
    # Use ImageMagick if available so the bundle has both formats; if not,
    # the LaTeX build will fall back to the PNG.
    converted = shutil.which("convert")
    if converted is not None:
        import subprocess
        subprocess.run([converted, str(out_png), str(out_pdf)],
                       check=False, capture_output=True)
        if out_pdf.exists():
            print(f"[case-study] wrote {out_pdf.relative_to(ROOT)} (via ImageMagick)")
    else:
        print("[case-study] note: ImageMagick `convert` not on PATH; "
              "skipping PDF export. PNG is sufficient for LaTeX include.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--uniprot", default="P00519", help="UniProt accession (default P00519, ABL1).")
    p.add_argument("--position", type=int, default=433, help="Residue position (1-indexed).")
    p.add_argument("--gene", default="ABL1", help="Gene symbol for the caption.")
    p.add_argument("--variant", default="A433T", help="Variant label for the caption.")
    p.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR)
    p.add_argument("--struct-features", type=Path, default=DEFAULT_STRUCT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--radius", type=float, default=8.0,
                   help="Spatial neighborhood radius (A) around the variant Calpha.")
    args = p.parse_args()

    render(args.uniprot, args.position, args.gene, args.variant,
           pdb_dir=args.pdb_dir, struct_features=args.struct_features,
           out_png=args.out, neighborhood_radius=args.radius)


if __name__ == "__main__":
    main()
