# daisyluAMR

Hold off until this is tested (10/6/17)

Installation
---

install git lfs (large file storage)
  follow instructions here
  after installation you should see something good for git lfs

check networks/models directory, if .weights files are not downloaded, issue
  git lfs fetch
  
download wikification

install tensorflow

install smatch 

modify system/daisylu_config.py to contain paths for your setup


Test
----

a set of files is provided, one with sentences and ids, the other with human generated AMR.

cd system
python daisylu_main.py

should result in a smatch score of around 0.6519 

sqlite, pandas, other python packages based on runtime errors




