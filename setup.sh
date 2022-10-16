#!/bin/bash

if { conda env list | grep 'anlp'; } >/dev/null 2>&1; 
then
    echo "Activating anlp conda environment"
    conda activate anlp
fi

pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm