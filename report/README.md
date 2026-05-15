# report/

Final report for pocket-VEP. NeurIPS 2024 style, natbib + bibtex.

## Build

```bash
cd report/
make            # builds main.pdf
make watch      # rebuilds on every save (latexmk -pvc)
make clean      # remove build artefacts, keep PDF
make distclean  # remove everything including the PDF
```

Requires `pdflatex` + `bibtex` + `latexmk` on PATH (the `texlive-latex-extra`
and `texlive-bibtex-extra` Debian/Ubuntu packages cover these).

## Layout

- `main.tex` — single-file source, sections one per top-level concern
- `refs.bib` — bibliography. Cite key convention is
  `firstauthorYEARshortname` (e.g. `frazer2021eve`)
- `figures/` — drop PDF/PNG figures here; `\graphicspath` already points
  at it so you can `\includegraphics{ablation_bars}` without the prefix
- `neurips_2024.sty` — official NeurIPS 2024 style file, vendored. Don't
  edit.

## Drafting workflow

The style is loaded with `[final]` for the camera-ready look (visible
author block). While drafting, switch to `[preprint]` if you want
something less aggressive on the title page:

```latex
\usepackage[preprint]{neurips_2024}
```
