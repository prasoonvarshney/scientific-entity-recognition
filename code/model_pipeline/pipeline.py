import copy
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import DatasetLoader
from constants import TEST_DATA_FILE_PATH, label2id

metric = evaluate.load("seqeval")

class TrainingPipeline: 
    def __init__(self, batch_size=8, lr=5e-5, n_epochs=10, weight_decay=1e-5, model_checkpoint='bert-base-cased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_checkpoint = model_checkpoint

        if self.model_checkpoint == "bert-base-cased":
            self.model_name = "bert_base"
        elif self.model_checkpoint == "microsoft/deberta-v3-base":
            self.model_name = "deberta"
        elif self.model_checkpoint == "allenai/scibert_scivocab_cased":
            self.model_name = "scibert"
        elif self.model_checkpoint == "KISTI-AI/scideberta-cs":
            self.model_name = "scideberta"
        else:
            self.mode_name = "others"

        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.num_labels = len(self.label2id.keys())

        print('Initializing tokenizer from {}'.format(model_checkpoint))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        if self.model_checkpoint == "KISTI-AI/scideberta-cs" or self.model_checkpoint == "microsoft/deberta-v3-base":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, add_prefix_space=True)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        print('Initializing model from {}...'.format(model_checkpoint))
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_checkpoint, 
            id2label=self.id2label, label2id=self.label2id, num_labels=self.num_labels
        )
        self.model.to(self.device)

        print('Loading dataset...')
        self.dataset = DatasetLoader().dataset
        self.label_names = self.dataset["train"].features["ner_tags"].features.names
        print(f"DEBUG!!!!!!!!!!! {self.label_names} {type(self.label_names)}")
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        self.train_data = DataLoader(self.tokenized_dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=self.data_collator)
        self.val_data = DataLoader(self.tokenized_dataset['validation'], batch_size=batch_size, shuffle=False, collate_fn=self.data_collator)
        self.test_data = DataLoader(self.tokenized_dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=self.data_collator)
        self.batch_size = batch_size
        self.args = TrainingArguments(
            "bert-finetuned-ner",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            num_train_epochs=n_epochs,
            weight_decay=weight_decay,
            push_to_hub=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )


    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=False, is_split_into_words=True
        )
        # tokenized_inputs["word_ids"] = tokenized_inputs.word_ids()
        all_labels = examples["ner_tags"]
        new_labels = []
        all_word_ids = []
        # print(all_labels)

        max_length = 0
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            if len(word_ids) > max_length:
                max_length = len(word_ids)

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            word_ids_corrected = [id if id is not None else -100 for id in word_ids]
            word_ids_corrected += [-100] * (max_length - len(word_ids_corrected))  # Explicitly pad word_ids here because huggingface internal classes only pad input_ids and labels
            all_word_ids.append(word_ids_corrected)
            # print(labels, word_ids)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        tokenized_inputs["word_ids"] = all_word_ids
        return tokenized_inputs

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX  # Pra: this is actually fine
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    def unalign_labels_by_word_ids(self, labels, word_ids):
        prev_id = -100
        padded_word_ids = []
        for i, word_id in enumerate(word_ids): 
            if prev_id == word_id:
                padded_word_ids.append(-100)
            else:
                padded_word_ids.append(word_id)
            prev_id = word_id

        assert len(labels) == len(padded_word_ids)

        unaligned_labels = []
        for label, padded_word_id in zip(labels, padded_word_ids):
            if padded_word_id != -100: 
                unaligned_labels.append(label)
        return unaligned_labels

    def generate_predictions(self, test_data=None):
        if test_data is None:
            test_data = self.test_data  # use our own test_data
            filename = "test_predictions_ours_{}.conll".format(self.model_name)
        else:
            filename = "test_predictions_{}.conll".format(self.model_name)  # test predictions from ANLP test set

        self.model.eval()
        true_labels = []
        pred_labels = []
        unaligned_pred_labels = []
        decoded_inputs = []
        all_original_spacy_tokens = []

        for i, batch in enumerate(tqdm(test_data)):
            with torch.no_grad():
                input_ids = batch['input_ids']
                word_ids = batch['word_ids']
                batch = { k: v.to(self.device) for k, v in batch.items() if k != 'word_ids'}
                outputs = self.model(**batch)
                decoded_inputs += (self.tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))
                
            s_lengths = batch['attention_mask'].sum(dim=1)
            for idx, length in enumerate(s_lengths):
                cur_true_labels = []
                cur_pred_labels = []
                cur_word_ids = word_ids[idx][:length]
                true_values = batch['labels'][idx][:length]
                pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]
                for true, pred in zip(true_values, pred_values):
                    true_label = true.item()
                    pred_label = pred.item()
                    cur_true_labels.append(true_label)
                    cur_pred_labels.append(pred_label)

                true_labels.append(cur_true_labels)
                pred_labels.append(cur_pred_labels)
                original_spacy_tokens = self.dataset["train"]['tokens'][i*self.batch_size+idx]
                unaligned_cur_pred_labels = self.unalign_labels_by_word_ids(
                    cur_pred_labels, 
                    cur_word_ids.tolist()
                )
                unaligned_pred_labels.append(unaligned_cur_pred_labels)
                all_original_spacy_tokens.append(original_spacy_tokens)
        
        print('Saving predictions to {}...'.format(TEST_DATA_FILE_PATH + filename))
        with open(TEST_DATA_FILE_PATH + filename, 'w') as output_file:
            for i in range(len(all_original_spacy_tokens)):
                for token, label in zip(all_original_spacy_tokens[i], unaligned_pred_labels[i]):
                    human_label = self.id2label[label]
                    output_file.write(token + "\t" + human_label + "\n")
                output_file.write("\n")


    ## Code below to generate confusion matrix
    def display_confusion_matrix(self, confusion_matrix):
        fig, ax = plt.subplots(figsize=(10, 10))
        confusion_matrix_normalized = copy.deepcopy(confusion_matrix).astype(float)
        num_labels = len(label2id.keys())
        # Normalize by dividing every row by its sum
        for i in range(num_labels):
            confusion_matrix_normalized[i] = confusion_matrix_normalized[i] / confusion_matrix_normalized[i].sum()
        ax.matshow(confusion_matrix_normalized)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='x-large')

        labels = list(label2id.keys())
        ids = np.arange(len(labels))

        ax.set_ylabel('True Labels', fontsize='x-large')
        ax.set_xlabel('Pred Labels', fontsize='x-large')

        ax.set_xticks(ids)
        ax.set_xticklabels(labels)

        ax.set_yticks(ids)
        ax.set_yticklabels(labels)

        fig.tight_layout()
        plt.show()


    def generate_confusion_matrix(self, val_data):
        self.model.eval()
        confusion = torch.zeros(self.num_labels+1, self.num_labels+1)
        for i, batch in enumerate(tqdm(val_data)):
            with torch.no_grad():
                batch = { k: v.to(self.device) for k, v in batch.items() if k != 'word_ids' }
                outputs = self.model(**batch)
                    
            s_lengths = batch['attention_mask'].sum(dim=1)
            for idx, length in enumerate(s_lengths):
                true_values = batch['labels'][idx][:length]
                pred_values = torch.argmax(outputs[1], dim=2)[idx][:length]
                for true, pred in zip(true_values, pred_values):
                    # insert padding tokens as label 10
                    true_label = true.item()
                    pred_label = pred.item()
                    if true_label == -100 or pred_label == -100:
                        continue
                    confusion[true_label][pred_label] += 1

        return confusion.numpy().astype(int)

