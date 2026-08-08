FIGS := $(patsubst %.tex,%.pdf,$(filter-out figures/palette.tex,$(wildcard figures/*.tex)))

.PHONY: all clean
all: model-draft-slides.pdf

model-draft-slides.pdf: model-draft-slides.tex $(FIGS)
	latexmk -pdf -interaction=nonstopmode $<

figures/%.pdf: figures/%.tex figures/palette.tex
	cd figures && pdflatex -interaction=nonstopmode -halt-on-error $(notdir $<)

clean:
	latexmk -C model-draft-slides.tex
	rm -f figures/*.aux figures/*.log figures/*.fls figures/*.fdb_latexmk
