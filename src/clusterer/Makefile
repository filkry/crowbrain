.phony: all backup tar

SRC_FILES = Makefile clustering_app.py feature_extraction.py import_data.py naive_bayes.py qt_app.py

all:
	pyside-uic clustering_app.ui > clustering_app.py
	#pyside-uic coverage.ui > coverage.py
	pyside-uic import_cluster_text.ui > import_cluster_text.py
	#pyside-uic similarity_selector.ui > similarity_selector.py
	#pyside-uic title.ui > title.py
backup:
	cbackup.py *py *ui
tar:
	tar cjf clusterer.tar.bz2 ${SRC_FILES}
