# Build the pocket-VEP report.
#
#   make            -> main.pdf
#   make watch      -> rebuild on save (latexmk -pvc)
#   make clean      -> remove build artefacts but keep main.pdf
#   make distclean  -> remove everything including main.pdf
#
# Uses pdflatex + bibtex (natbib).

MAIN := main
LATEXMK := latexmk
# -halt-on-error left off so the build still produces a PDF when only
# warnings (e.g. undefined citations) are present.
LATEXMK_FLAGS := -pdf -bibtex -interaction=nonstopmode -file-line-error

.PHONY: all watch clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib neurips_2024.sty
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN).tex

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml

distclean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml
