MAMAN ?= maman13
NOTEBOOK := $(MAMAN).ipynb
STYLE_FILE ?= pdf-style.yml

.PHONY: all latex notebook quarto merge clean

all: notebook quarto

notebook:
	jupyter nbconvert --to notebook --execute ./$(NOTEBOOK) --inplace

quarto: notebook
	quarto render $(NOTEBOOK) --metadata-file $(STYLE_FILE) --to pdf

clean:
	jupyter nbconvert --clear-output --inplace $(NOTEBOOK)
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz