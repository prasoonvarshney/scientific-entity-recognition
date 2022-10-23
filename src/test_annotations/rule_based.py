import json
import re
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="bert.json")
args = parser.parse_args()
source = args.source

MAPPING_FILE = 'reverse_mapping.pkl'
DATA_FILE = source
OUTPUT_FILE = DATA_FILE.split('.json')[0] + '_edited.json'

def generate_id():
    alphanum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    res = ''.join((random.choice(alphanum) for x in range(10)))
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
    results = annotations[0]['result']

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
                    "labels": [reverse_mapping[word]]
                },
                "id": random_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual"
            }

            item['annotations'][0]['result'].append(new_object)
            print(f"Start: {start}, end: {end}, cased_word: {cased_word}")
            print(f"New object: {new_object}")
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