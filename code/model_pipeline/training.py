import numpy as np
import evaluate

from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    DataLoader, TrainingArguments, Trainer

from dataloader import DatasetLoader
from constants import label2id

metric = evaluate.load("seqeval")


class TrainingPipeline: 
    def __init__(self, device, BATCH_SIZE, LR, N_EPOCHS, WEIGHT_DECAY):
        self.model_checkpoint = "bert-base-cased"
        id2label = {v: k for k, v in label2id.items()}
        self.label_names = label2id.keys()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_checkpoint, id2label=id2label, label2id=label2id, num_labels=len(label2id.keys()))
        self.model.to(device)

        self.dataset = DatasetLoader().dataset
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )

        self.train_data = DataLoader(self.tokenized_dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.data_collator)
        self.val_data = DataLoader(self.tokenized_dataset['validation'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=self.data_collator)
        self.test_data = DataLoader(self.tokenized_dataset['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=self.data_collator)
       
        self.args = TrainingArguments(
            "bert-finetuned-ner",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LR,
            num_train_epochs=N_EPOCHS,
            weight_decay=WEIGHT_DECAY,
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

    def unalign_labels_by_word_ids(labels, word_ids):
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