import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="bert.json")
args = parser.parse_args()
source = args.source  # annotation file

mapping = {}
labels = 'MethodName HyperparameterName HyperparameterValue DatasetName MetricValue MetricName TaskName'.split()

for label in labels:
    mapping[label] = set()

with open(source, 'r') as file:
    data = json.load(file)

with open(source, 'w') as file:
    json.dump(data, file, indent=4)  # redump the just loaded data with indent 4 for readability and ease of comparison with edited.json file which is also exported with indent 4

for item in data:
    annotations = item['annotations']
    if len(annotations) == 0:
        continue  # don't look at the sentences yet to be annotated

    assert len(annotations) == 1

    results = annotations[0]['result']
    
    for result in results:
        text = result['value']['text'].lower()
        if len(text) > 1:
            label = result['value']['labels'][0]
            mapping[label].add(text)  # label to text

for label in mapping: 
    mapping[label] = list(mapping[label])

# OUTPUT
# {'MethodName': {'BERT',
#   'BERT BASE',
#   'BERT LARGE',
#   'BERTBASE',
#   'BERTLARGE',
#   'BiLSTM',
#   'Bidirectional Encoder Representations from Transformers',
# ...

reverse_mapping = {}
for label in mapping:
    for item in mapping[label]:
        reverse_mapping[item] = label  # text to label

# OUTPUT
# {'LSTMs': 'MethodName',
#  'denoising auto - encoders': 'MethodName',
#  'accuracy': 'MetricName',
# ...
# Save outputs to pickle
with open('mapping.json', 'w') as f:
    json.dump(mapping, f, indent=4)

with open('reverse_mapping.json', 'w') as f:
    json.dump(reverse_mapping, f, indent=4)
