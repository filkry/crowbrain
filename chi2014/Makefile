view: all
	evince paper.pdf &

all:
	pdflatex paper.tex
	bibtex paper.aux
	pdflatex paper.tex
	pdflatex paper.tex

sample:
	pdflatex sigchi-sample.tex
	bibtex sigchi-sample.aux
	pdflatex sigchi-sample.tex
	pdflatex sigchi-sample.tex

clean:
	-rm paper.pdf *.aux *.log *.bbl *.blg *.lof *.lot *.out *.toc
