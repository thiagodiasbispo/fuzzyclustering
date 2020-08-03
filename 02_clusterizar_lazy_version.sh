#!/bin/bash

NB_NAME=02_clusterizar_lazy_version.ipynb
SCRIPT_NAME=02_clusterizar_lazy_version.automatically_exported

jupyter nbconvert --output-dir src --output $SCRIPT_NAME --to script $NB_NAME

python src/$SCRIPT_NAME.py data