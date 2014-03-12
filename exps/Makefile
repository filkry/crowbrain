thesis:
	python idea_forest_descriptive.py
	python gen_forest_figures.py
	python gen_appendices.py
	python simulate_error_tree.py
	python rate_model.py
	python split_originality_blend_model.py
	python siam_change_prob.py
	python siam_time.py
	cp figures/* ~/enc_projects/thesis/tex/
	cp tex/* ~/enc_projects/thesis/tex/

export:
	ipython nbconvert --to html $(nb)

export_tex:
	ipython nbconvert --to latex $(nb)
