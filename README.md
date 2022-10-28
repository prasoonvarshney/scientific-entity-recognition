# Advanced NLP: From Scratch: Fall 2022
[nlp-from-scratch-assignment](https://github.com/neubig/nlp-from-scratch-assignment-2022/) for 11-711, Advanced NLP course at Carnegie Mellon University.

## Run Instructions
To run the model training and evaluation pipeline, cd into the root folder of the repository and run:

`python code/model_pipeline/sciner.py --model-checkpoint KISTI-AI/scideberta-cs --lr 5e-5 --epochs 10 --weight_decay 1e-6 --batch_size 8`

# Assignment Description
This objective is to perform recognition of scientific entities in research papers. 
This repository contains all required components to do the task: 
1. Scraping scripts to fetch research paper PDFs at code/data_collection
2. Parsing scripts to parse the PDFs at code/data_collection
3. A collection of 32 manually annotated papers (gold-standard) at data/annotated
    a. Scripts to split and create train and dev sets are located at data/created_data_train_test_splits
    b. Held out test set is located at data/test
4. Model training pipeline that achieves 0.626 F1 on the held-out set.


## Directory Tree Structure

advanced-nlp-f22-hw2
├── code
│   ├── data_collection
│   │   ├── annotation_scripts
│   │   │   ├── bert.conll
│   │   │   ├── bert_copy.json
│   │   │   ├── bert_edited.json
│   │   │   ├── bert.json
│   │   │   ├── bert_min.json
│   │   │   ├── generate_mapping.py
│   │   │   ├── lda_edited.json
│   │   │   ├── lda.json
│   │   │   ├── mapping.json
│   │   │   ├── reverse_mapping.json
│   │   │   └── rule_based.py
│   │   ├── constants.py
│   │   ├── example_scipdf_output.json
│   │   ├── labelstudio_collector.py
│   │   ├── parser.py
│   │   ├── pdf_urls.txt
│   │   ├── random_pdf.py
│   │   ├── sampled_urls.txt
│   │   ├── scrape_all.py
│   │   └── scrape.py
│   ├── model_pipeline
│   │   ├── constants.py
│   │   ├── dataloader.py
│   │   ├── pipeline.py
│   │   └── sciner.py
│   └── notebooks
│       ├── ANLP_NER_BertyBoy.ipynb
│       ├── ANLP_NER_deberta.ipynb
│       ├── ANLP_NER_SCIBERT.ipynb
│       ├── ANLP_NER_scideberta.ipynb
│       └── NER_Model_Pipeline.ipynb
├── data
│   ├── annotated
|   |       (our annotated conll files by paper as exported from Label Studio)
│   ├── created_data_train_test_splits
|   |       (our annotations split into train.conll and test.conll)
│   ├── parsed_pdfs
|   |       (txt files of parsed papers)
│   ├── summary_of_parsed_files.json
│   ├── test
|   |       (held out test set files and our model predictions on them)
├── github.txt
├── LICENSE
├── README.md
├── requirements.txt
└── setup.sh