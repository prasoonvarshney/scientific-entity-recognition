import json
import pickle

mapping = {}
labels = 'MethodName HyperparameterName HyperparameterValue DatasetName MetricValue MetricName TaskName'.split()

for label in labels:
    mapping[label] = set()

with open('bert.json', 'r') as file:
    data = json.load(file)

for item in data:
    annotations = item['annotations']
    assert len(annotations) == 1

    results = annotations[0]['result']
    
    for result in results:
        text = result['value']['text'].lower()
        if len(text) > 1:
            label = result['value']['labels'][0]
            mapping[label].add(text)  # label to text

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
f = open('mapping.pkl', 'wb')
pickle.dump(mapping, f)

f = open('reverse_mapping.pkl', 'wb')
pickle.dump(reverse_mapping, f)
