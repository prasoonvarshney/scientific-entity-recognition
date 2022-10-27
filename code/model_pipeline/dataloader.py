import requests
import os
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Sequence
import pandas as pd

from constants import LABEL_STUDIO_URL, LABEL_STUDIO_AUTH_HEADER, \
    TRAIN_PAPERS, TEST_PAPERS, TRAIN_DATA_FILE_PATH, TEST_DATA_FILE_PATH, TEST_DATA_SENTENCES_FILE, \
    label2id


class DatasetLoader: 
    def __init__(self):
        class_names = list(label2id.keys())
        features = Features({
            'id': Value(dtype='string'),
            'tokens': Sequence(Value(dtype='string')),
            'ner_tags': Sequence(ClassLabel(names=class_names))
        })
        test_features = Features({
            'id': Value(dtype='string'),
            'tokens': Sequence(Value(dtype='string')),
            'ner_tags': Sequence(ClassLabel(names=class_names))
        })

        print(f"Loading train and validation datasets...")
        train_annotations = collect_train_test_data(split="train")
        val_annotations = collect_train_test_data(split="test")
        print(f"Loading test dataset...")
        test_data = collect_held_out_test_data()

        train_df = pd.DataFrame(train_annotations, columns=["tokens", "ner_tags"]).reset_index().rename(columns={'index': 'id'})
        val_df = pd.DataFrame(val_annotations, columns=["tokens", "ner_tags"]).reset_index().rename(columns={'index': 'id'})
        test_df = pd.DataFrame(test_data, columns=["tokens", "ner_tags"]).reset_index().rename(columns={'index': 'id'})

        train_dataset = Dataset.from_pandas(train_df, features=features)
        val_dataset = Dataset.from_pandas(val_df, features=features)
        test_dataset = Dataset.from_pandas(test_df, features=test_features)
        self.dataset = DatasetDict({
            "train": train_dataset, 
            "validation": val_dataset, 
            "test": test_dataset
        })
        print(f"Datasets done...")



def collect_train_test_data(split = "train"): 
    if split not in ["train", "test"]: 
        return None

    with open(TRAIN_DATA_FILE_PATH + split + ".conll", 'r') as f:
        train_annotations_from_file = f.readlines()

    sentences = []
    tokens = []
    labels = []
    for line in train_annotations_from_file:
        if line == "\n":
            # end of sentence
            if len(tokens): 
                sentences.append({"tokens": tokens, "ner_tags": labels})
            tokens = []
            labels = []
        else: 
            token, label = line.split('\t')
            tokens.append(token)
            labels.append(label.replace('\n', ''))
    return sentences


def collect_held_out_test_data(): 
    with open(os.path.join(TEST_DATA_FILE_PATH, TEST_DATA_SENTENCES_FILE), 'r') as test_file:
        test_data = test_file.readlines()

    test_data = [{"tokens": sentence.split(), "ner_tags": [0 for x in sentence.split()]} for sentence in test_data]
    return test_data
