import json
import pickle
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="bert.json")
args = parser.parse_args()
source = args.source

MAPPING_FILE = 'reverse_mapping.pkl'
DATA_FILE = source
OUTPUT_FILE = DATA_FILE.split('.')[0] + '_edited.json'

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


reverse_mapping = pickle.load(open('reverse_mapping.pkl', 'rb'))
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
        annotated_words_and_labels[result['value']['text']] = result['value']['labels']

    for word in reverse_mapping:
        if word in text.lower():
            start = text.lower().index(word)
            end = start + len(word)
            cased_word = text[start:end]  # to be added in JSON

            # word is not annotated
            if cased_word not in annotated_words_and_labels:
                # to check word is not within another word, e.g. general has ner
                
                # start of sentence
                # bert is a transformer --> true 
                # bertrand et. al --> false
                if start == 0 and (end) < len(text) and (text[end].isalnum() == False):
                    annotate = True 

                # middle of sentence
                # the ner task --> true
                # generally --> false
                elif start > 0 and (end) < len(text) and (text[start-1].isalnum() == False) and (text[end].isalnum() == False):
                    annotate = True

                # end of sentence
                # the dataset we are using is mnli --> true
                # my name is himnli --> false
                elif start > 0 and (end) == len(text) and (text[start-1].isalnum() == False):
                    annotate = True

                else:
                    annotate = False

                if annotate:
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
            
            # commented out because we just one one annotation per word block
            # missing annotation, or wrong annotation
            # if cased_word in annotated_words_and_labels and reverse_mapping[word] not in annotated_words_and_labels[cased_word]:
            #     for i, result in enumerate(item['annotations'][0]['result']):
            #         if result['value']['text'] == cased_word:
            #              item['annotations'][0]['result'][i]['value']['labels'].append(reverse_mapping[word])

            break  # no need to check for shorter words, e.g. BERT BASE preferred over BERT

# save new json
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f)