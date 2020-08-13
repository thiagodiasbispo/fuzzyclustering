#!/bin/bash

NB_NAME=02_clusterizar_lazy_version.ipynb
SCRIPT_NAME=02_clusterizar_lazy_version.exportado_automaticamente

jupyter nbconvert --output-dir src --output $SCRIPT_NAME --to script $NB_NAME

git add src/*
git commit -m "Nova versão de src/$SCRIPT_NAME.py"

python src/$SCRIPT_NAME.py data