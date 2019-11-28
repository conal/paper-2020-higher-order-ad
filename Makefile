paper = hoad

.PRECIOUS: %.tex %.pdf

all: $(paper).pdf

see: $(paper).see

dots = $(wildcard Figures/*.dot)
pdfs = $(addsuffix .pdf, $(basename $(dots)))

dots: $(dots)
pdfs: $(pdfs)

latex=latexmk -pdf -halt-on-error

%.pdf: %.tex $(pdfs) bib.bib Makefile
	$(latex) $*.tex

%.tex: %.lhs macros.tex formatting.fmt Makefile
	lhs2TeX -o $*.tex $*.lhs

showpdf = open -a Skim.app

%.see: %.pdf
	${showpdf} $*.pdf

# Cap the size so that LaTeX doesn't choke.
%.pdf: %.dot # Makefile
	dot -Tpdf -Gmargin=0 -Gsize=10,10 $< -o $@

pdfs: $(pdfs)

clean:
	rm -f $(paper)*.{tex,pdf,aux,nav,snm,ptb,log,out,toc,bbl,blg,fdb_latexmk,fls}

# Handy, e.g., with "make push web"
push:
	git push

STASH=conal@conal.net:/home/conal/web/papers/higher-order-ad

web: web-token

web-token: $(paper).pdf
	scp $? $(STASH)/higher-order-ad.pdf
	touch $@
