.phony: all backup tar

SRC_FILES = Makefile clustering_app.py feature_extraction.py import_data.py naive_bayes.py qt_app.py

all:
	pyside-uic clustering_app.ui > clustering_app.py
backup:
	cbackup.py *py *ui
tar:
	tar cjf clusterer.tar.bz2 ${SRC_FILES}
