### GENERATE MAPPING

import re
import json
import argparse
import random

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


### ANNOTATE AND SAVE
MAPPING_FILE = 'reverse_mapping.pkl'
DATA_FILE = source
OUTPUT_FILE = DATA_FILE.split('.json')[0] + '_edited.json'

def generate_id():
    alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    res = ''.join((random.choice(alphanum) for x in range(10)))
    return res

def generate_num_id(item_id):
    half = item_id//2
    res = random.choice([i for i in range(half,item_id)])
    return res

def fill(to_fill, start, end, text, label):
    to_fill['value']['start'] = start
    to_fill['value']['end'] = end
    to_fill['value']['text'] = text
    to_fill['value']['labels'].append(label)
    to_fill['id'] = generate_id()

with open('reverse_mapping.json', 'r') as f:
    reverse_mapping = json.load(f)
rev_keys = sorted(list(reverse_mapping.keys()), key=lambda x: len(x), reverse=True)

with open(DATA_FILE, 'r') as file:
    data = json.load(file)

for item in data:

    text = item['data']['text']
    lowercase_text = text.lower()
    annotations = item['annotations']
    if len(annotations) > 0:
        results = annotations[0]['result']
    else: 
        results = []

    annotated_words_and_labels = {}

    for result in results:
        annotated_words_and_labels[
            (result['value']['text'], result['value']['start'], result['value']['end'])
        ] = result['value']['labels']

    # build the right regular expression with all longest to shortest words in order
    regexps = []
    for word in rev_keys: 
        regexps.append(r'\b' + word + r'\b')
    regexp = "|".join(regexps)

    for match_object in re.finditer(regexp, text, flags=re.IGNORECASE):
        # fyi re.finditer returns an iterator over all non-overlapping matches of the regular expression
        start = match_object.span()[0]
        end = match_object.span()[1]
        cased_word = text[start:end]  # to be added in JSON

        # word is not already annotated
        if (cased_word, start, end) not in annotated_words_and_labels:
            # to check word is not within another word, e.g. general has ner
            # start of sentence
            # bert is a transformer --> true 
            # bertrand et. al --> false
            # middle of sentence
            # the ner task --> true
            # generally --> false
            # end of sentence
            # the dataset we are using is mnli --> true
            #  --> false
            # UPDATE: ALL these cases are handled simply by using "word boundaries" (\b) in regular expressions
            # add instance in annotations
            random_id = generate_id()
            new_object = {
                "value": {
                    "start": start,
                    "end": end,
                    "text": cased_word,
                    "labels": [reverse_mapping[cased_word.lower()]]
                },
                "id": random_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
            }

            if len(item['annotations']) == 0:
                num_id = generate_num_id(item['id'])
                item['annotations'] = [{"id": num_id, "completed_by": 4, 'result': []}]
            item['annotations'][0]['result'].append(new_object)
            # print(f"Start: {start}, end: {end}, cased_word: {cased_word}")
            # print(f"New object: {new_object}")
        # commented out because we just one one annotation per word block
        # missing annotation, or wrong annotation
        # if cased_word in annotated_words_and_labels and reverse_mapping[word] not in annotated_words_and_labels[cased_word]:
        #     for i, result in enumerate(item['annotations'][0]['result']):
        #         if result['value']['text'] == cased_word:
        #              item['annotations'][0]['result'][i]['value']['labels'].append(reverse_mapping[word])
        # no need to check for shorter words, e.g. BERT BASE preferred over BERT

# save new json
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=4)