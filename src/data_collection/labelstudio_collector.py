import os
import requests

LABEL_STUDIO_URL = "http://ec2-54-175-11-3.compute-1.amazonaws.com:8080"
LABEL_STUDIO_AUTH_HEADER = {'Authorization': 'Token 8c7f4625665248b9ee072b34791137cdc6fcaf18'}

PROJECT_ID_TO_PAPER_MAPPING = {
    "9": "P15-2083", 
    "11": "P18-2051", 
    "12": "P16-2007", 
    "13": "2022.coling-1.308", 
    "16": "P17-2066", 
    "18": "2020.coling-main.130", 
    "20": "2021.semeval-1.12", 
    "22": "P19-1020", 
    "23": "2022.acl-long.576", 
    "24": "2021.eacl-main.224", 
    "25": "2021.wanlp-1.15", 
    "27": "2020.emnlp-main.563", 
    "29": "P19-1128", 
    "31": "P15-1070", 
    "29": "P19-1128", 
    "29": "P19-1128", 
    "32": "P15-2085", 
    "33": "P16-1158", 
    "34": "P17-1038", 
    "35": "P18-1085",
    "36": "P18-1182",
    "37": "2020.findings-emnlp.20", 
    "40": "2020.acl-main.169",
    "41": "2020.aacl-main.32",
    "42": "2021.findings-acl.175", 
    "43": "P18-3010",
    "44": "2020.loresmt-1.5", 
    "49": "P16-2056", 
    "51": "2020.ecnlp-1.8", 
    "52": "2022.acl-short.36",
    
}

FILE_PATH = "./src/data_collection/annotated/"


def convert_label_studio_output_to_standard_conll_format(label_studio_conll_output): 
    split_ls_output = label_studio_conll_output.split('\n')
    rectified_split_output = []

    for line in split_ls_output:
        words = line.split(' ')

        if len(words) <= 1:
            # end of sentence 
            rectified_split_output.append(line)

        elif len(words) == 3 and words[0] == '-DOCSTART-':
            # start of doucument - IGNORE
            pass

        elif len(words) == 4:
            # here we go
            rectified_split_output.append("\t".join([words[0], words[3]]))
        else: 
            # unexpected, handle
            raise Exception(f"Unexpected format in CoNNL format output line: {words}")

    standard_conll_output = '\n'.join(rectified_split_output)
    return standard_conll_output


def parse_label_studio_output_for_modeling(label_studio_conll_format_output): 
    annotated = label_studio_conll_format_output.split('\n')
    annotations_parsed_per_document = []
    annotations_parsed_per_sentence = {'tokens': [], 'ner_tags': []}

    for line in annotated:
        words = line.split(' ')
        # print(f"line: {line}, words: {words}")
        if len(words) <= 1:
            # end of sentence 
            if len(annotations_parsed_per_sentence['tokens']):
                annotations_parsed_per_document.append(annotations_parsed_per_sentence)
            annotations_parsed_per_sentence = {'tokens': [], 'ner_tags': []}
        elif len(words) == 3 and words[0] == '-DOCSTART-':
            # start of doucument 
            pass
        elif len(words) == 4:
            # here we go
            annotations_parsed_per_sentence['tokens'].append(words[0])
            annotations_parsed_per_sentence['ner_tags'].append(words[3])
        else: 
            # unexpected, handle
            raise Exception(f"Unexpected format in CoNNL format output line: {words}")

    annotations_parsed_per_document = list(reversed(annotations_parsed_per_document))

    for sentence in annotations_parsed_per_document:
        print(f"{sentence['tokens']}\n{sentence['ner_tags']}\n\n")
    
    return annotations_parsed_per_document


def collect_annotated_files_from_label_studio():
    for project_id in PROJECT_ID_TO_PAPER_MAPPING.keys():
        url = LABEL_STUDIO_URL + "/api/projects/" + project_id + "/export?exportType=CONLL2003&download_all_tasks=true"
        resp = requests.get(url, headers=LABEL_STUDIO_AUTH_HEADER)
        if resp.status_code != 200:
            continue

        labeled_document = convert_label_studio_output_to_standard_conll_format(resp.text)
        with open(os.path.join(FILE_PATH, PROJECT_ID_TO_PAPER_MAPPING[project_id] + ".conll"), 'w') as f:
            f.write(labeled_document)


if __name__ == "__main__":
    collect_annotated_files_from_label_studio()