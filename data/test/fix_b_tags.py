CONLL_FILE = "test_predictions_scibert_para.conll"
OUTPUT_CONLL_FILE = "test_predictions_scibert_para_fixed.conll"

output_tags = []

previous_ner_tag = None

with open(CONLL_FILE, 'r') as f:
    for line in f.readlines():
        if line == '\n':
            previous_ner_tag = None
            output_tags.append(line)
        else:
            text, tag = (line.strip().split('\t'))
            
            if tag == 'O':
                pos = 'O'
                ner = 'O'
            else:
                pos, ner = tag.split('-')
            
            # this should be B, not I
            if previous_ner_tag != ner and pos == 'I':
                tag = 'B-' + ner
                previous_ner_tag = ner
                output_tags.append((text, tag))

            # otherwise
            else:
                previous_ner_tag = ner
                output_tags.append((text, tag))


with open(OUTPUT_CONLL_FILE, 'w') as f:
    for tag in output_tags:
        if tag == '\n':
            f.write(tag)
            # f.write('\n')
    
        else:
            pos, ner = tag[0], tag[1]
            f.write(pos+'\t'+ner)
            f.write('\n')

