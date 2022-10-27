import os
import requests

from constants import LABEL_STUDIO_URL, LABEL_STUDIO_AUTH_HEADER, TRAIN_PAPERS, TEST_PAPERS, ANNOTATIONS_FILE_PATH

PROJECT_ID_TO_PAPER_MAPPING = TRAIN_PAPERS | TEST_PAPERS

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

    annotations_parsed_per_document = list(annotations_parsed_per_document)

    # for sentence in annotations_parsed_per_document:
    #     print(f"{sentence['tokens']}\n{sentence['ner_tags']}\n\n")
    
    return annotations_parsed_per_document


def collect_annotated_files_from_label_studio():
    for project_id in PROJECT_ID_TO_PAPER_MAPPING.keys():
        url = LABEL_STUDIO_URL + "/api/projects/" + project_id + "/export"
        resp = requests.get(url, headers=LABEL_STUDIO_AUTH_HEADER, params={"exportType": "CONLL2003", "download_all_tasks": "true"})
        if resp.status_code != 200:
            continue

        labeled_document = convert_label_studio_output_to_standard_conll_format(resp.text)
        with open(os.path.join(ANNOTATIONS_FILE_PATH, PROJECT_ID_TO_PAPER_MAPPING[project_id] + ".conll"), 'w') as f:
            f.write(labeled_document)


def generate_train_test_split_data_files(split = "train"):
    if split == "train": 
        PROJECT_ID_TO_PAPER_MAPPING = TRAIN_PAPERS
    elif split == "test" or split == "validation": 
        PROJECT_ID_TO_PAPER_MAPPING = TEST_PAPERS
    else: 
        raise Exception("Split not recognized")

    all_labeled_documents_for_split = []
    for project_id in PROJECT_ID_TO_PAPER_MAPPING.keys():
        url = LABEL_STUDIO_URL + "/api/projects/" + project_id + "/export?exportType=CONLL2003&download_all_tasks=true"
        resp = requests.get(url, headers=LABEL_STUDIO_AUTH_HEADER)
        if resp.status_code != 200:
            continue

        labeled_document = convert_label_studio_output_to_standard_conll_format(resp.text)
        all_labeled_documents_for_split.append(labeled_document)

    with open(os.path.join(FILE_PATH, split + ".conll"), 'w') as f:
        for labeled_document in all_labeled_documents_for_split:
            f.write(labeled_document)


if __name__ == "__main__":
    collect_annotated_files_from_label_studio()
    generate_train_test_split_data_files()