import copy
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def display_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(10, 10))
    confusion_matrix_normalized = copy.deepcopy(confusion_matrix).astype(float)
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


def evaluate(model, val_data, num_labels, device):
    model.eval()
    confusion = torch.zeros(num_labels+1, num_labels+1)
    for i, batch in enumerate(tqdm(val_data)):
        with torch.no_grad():
            batch = { k: v.to(device) for k, v in batch.items() if k != 'word_ids' }
            outputs = model(**batch)
                
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

