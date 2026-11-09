.PHONY: all latex notebook quarto merge clean

all:  notebook quarto

notebook:
	jupyter nbconvert --to notebook --execute ./maman12.ipynb --inplace

quarto: notebook
	quarto render maman12.ipynb --metadata-file pdf-style.yml --to pdf

clean:
	jupyter nbconvert --clear-output --inplace maman12.ipynb
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz