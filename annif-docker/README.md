# Annif in Docker Test
* Unzip file to /tmp on your local machine (or wherever)
* Start annif in docker, bind to local project directory, in this case /tmp.
  `docker run -v /tmp/annif:/annif-projects -u $(id -u):$(id -g) -it quay.io/natlibfi/annif bash`
* load the vocabulary tsv
  `$ annif load-vocab uc-vocab ./uc-vocab.tsv --language en`
* train on dataset
  `$ annif train uc-en training-data/` -- no wildcard, just point to directory

### Notes 

* Needed to manually remove newlines from all of the .txt files
* Vocab needed 'http' protocol prepended to all urls in uc-vocab.tsv and .tsv for each work.
