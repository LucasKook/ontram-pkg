# Makefile for R Journal paper 

RNW = models-and-likelihood.Rnw
MAN = manuscript.Rnw

vignette:
	Rscript -e "library(knitr); knit2pdf('$(RNW)')"

manuscript:
	Rscript -e "library(knitr); knit2pdf('$(MAN)')"

clean:
	rm -f *~ *.log *.aux *.out Rplots.pdf \
	  *.synctex.gz .RData .Rhistory *-concordance*.tex \
	  *.bbl *.nav *.toc *.snm *.vrb \
	  *.blg *.brf $(RNW:.Rnw=.tex)

cleanall: clean
	rm -rf cache/ files/ figure/

all: vignette manuscript clean

