Marcus' Gecko Image Analysis protocol

- You need python 3.10.11 or later installed.
- Download the master branch from gitlab.com/cytrynlab/gecko-image-processing, place the zip file wherever you plan to use the code and unzip it.
- Command line: inside the folder TemplateMatching, run
  > > python3 -m venv venv
  > > pip install -r requirements.txt
- You might get a "notice" saying you should update pip. If so: >> pip install --upgrade pip
  and then install requirements again: >> pip install -r requirements.txt
  That's the initial one-time startup procedure. Now each time you start a new session (i.e. open up a command window and start running main.py), do the following:
- Inside the folder TemplateMatching, run
  > > source venv/bin/activate
- This sets the up a virtual environment for running python scripts.
- Remove/delete existing raw image files the subfolder img/ already in TemplateMatching and place the raw images for a new animal in that same folder.
- To run the full code on the image files, use
  > > python main.py
- To run just a single or multiple functions, run
  > > python main.py function1 function2
  > > where "function1/2" is replaced by the function(s) you want to run (options: match, manual, fitproj, format)
