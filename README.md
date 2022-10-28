# Advanced NLP: From Scratch: Fall 2022
[nlp-from-scratch-assignment](https://github.com/neubig/nlp-from-scratch-assignment-2022/) for 11-711, Advanced NLP course at Carnegie Mellon University.

## Run Instructions
To run the model training and evaluation pipeline, cd into the root folder of the repository and run:
`python code/model_pipeline/training.py --model-checkpoint microsoft/deberta-v3-base --lr 5e-5 --epochs 10 --weight_decay 1e-5`

# Assignment Description
This objective is to perform recognition of scientific entities in research papers. 
This repository contains all required components to do the task: 
1. Scraping scripts to fetch research paper PDFs at code/data_collection
2. Parsing scripts to parse the PDFs at code/data_collection
3. A collection of 32 manually annotated papers (gold-standard) at data/annotated
    a. Scripts to split and create train and dev sets are located at data/created_data_train_test_splits
    b. Held out test set is located at data/test
4. Model training pipeline that achieves 0.625 F1 on the held-out set.


