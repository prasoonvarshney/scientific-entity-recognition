import json
import lxml
import os
import re
import spacy
import scipdf
import random
import logging
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

FILE_PATH = "./src/data_collection/parsed_pdfs/"

def preprocess(line):
    # print(f"Line: {line}")
    line = line.strip()
    # line = re.sub(r';', '; ', line)
    line = re.sub(r'[\n\r]', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)

    if len(line) < 2: 
        return None
    tokenized = [token.text for token in nlp(line)]
    # print(f"Processed: {' '.join(tokenized)}")
    return " ".join(tokenized) + "\n"

def add_stop(text): 
    if text.endswith('.'): 
        return text
    else: 
        return text + '.'

def parse_pdf(pdf_url, reparse_parsed_files=False): 
    file_name = os.path.join(FILE_PATH, pdf_url.split('/')[-1] + ".txt")

    if os.path.exists(file_name) and not reparse_parsed_files:
        # If paper already parsed in a previous run, skip re-parsing, but return parsed file name
        parsed_article_dict = {
            "url": pdf_url,
            "file_name": file_name
        }
        return parsed_article_dict

    try:
        article = scipdf.parse_pdf_to_dict(pdf_url, as_list=False)
    except Exception as e:
        logging.error(f"Failed to parse PDF {pdf_url} with exception {e}")
        return None

    full_text = " ".join([add_stop(article['title']), add_stop(article['abstract'])])
    for section in article['sections']: 
        full_text = " ".join([full_text, add_stop(section['heading']), add_stop(section['text'])])

    # print(full_text)
    sentence_splitting_regex = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    
    full_text_tokenized = [preprocess(line) for line in re.split(sentence_splitting_regex, full_text)]
    full_text_tokenized = [text for text in full_text_tokenized if text is not None]

    with open(file_name, 'w') as f: 
        f.writelines(full_text_tokenized)

    parsed_article_dict = {
        "url": pdf_url,
        "file_name": file_name
    }
    return parsed_article_dict


if __name__ == "__main__":
    with open("./src/data_collection/sampled_urls.txt", 'r') as f:
        sampled_urls = f.readlines()
 
    # sampled_urls = random.choices(list(filter(lambda x: "aclanthology" in x, all_urls)), k=500)
    # year_list = ["aclanthology.org/2020", "aclanthology.org/2021", "aclanthology.org/2022", "aclanthology.org/P19", 
    #     "aclanthology.org/P18", "aclanthology.org/P17", "aclanthology.org/P16", "aclanthology.org/P15"]

    # sampled_urls = []
    # for year in year_list:
    #     sampled_urls_by_year = random.choices(list(filter(lambda x: year in x, all_urls)), k=5)
    #     sampled_urls = sampled_urls + sampled_urls_by_year

    all_parsed_pdfs = []
    for url in tqdm(sampled_urls): 
        url = re.sub(r'\n', '', url)
        parsed_pdf = parse_pdf(url)
        if parsed_pdf is not None: 
            all_parsed_pdfs.append(parsed_pdf)

    # Save down summary of URLs and saved processed files
    summary = {"parsed_pdfs": all_parsed_pdfs}
    summary_json = json.dumps(summary, indent=4)
    with open(os.path.join(FILE_PATH, "summary.json"), 'w') as f: 
        f.write(summary_json)