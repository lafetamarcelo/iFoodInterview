

html_note:
	rm -r ./docs/*.ipynb
	cp XGBoost\ Simple\ Classifier.ipynb ./docs
	cp Deep\ Networks.ipynb ./docs
	cp Support\ Vector\ Machines.ipynb ./docs
	cp Kohonen\ Maps.ipynb ./docs

clean_folder:
	rm *.h5