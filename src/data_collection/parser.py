import json
import os
import re
import spacy
import scipdf

nlp = spacy.load("en_core_web_sm")

FILE_PATH = "./src/data_collection/data/"

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

def parse_pdf(pdf_url): 
    article = scipdf.parse_pdf_to_dict(pdf_url, as_list=False)

    full_text = " ".join([add_stop(article['title']), add_stop(article['abstract'])])
    for section in article['sections']: 
        full_text = " ".join([full_text, add_stop(section['heading']), add_stop(section['text'])])

    # print(full_text)
    sentence_splitting_regex = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    
    full_text_tokenized = [preprocess(line) for line in re.split(sentence_splitting_regex, full_text)]
    full_text_tokenized = [text for text in full_text_tokenized if text is not None]

    file_name = os.path.join(FILE_PATH, url.split('/')[-1] + ".txt")

    with open(file_name, 'w') as f: 
        f.writelines(full_text_tokenized)

    parsed_article_dict = {
        "url": url,
        "file_name": file_name
    }
    return parsed_article_dict


if __name__ == "__main__":
    all_urls = ['https://aclanthology.org/2022.acl-long.1.pdf', 'https://aclanthology.org/N19-1423.pdf']

    all_parsed_pdfs = []
    for url in all_urls: 
        parsed_pdf = parse_pdf(url)
        all_parsed_pdfs.append(parsed_pdf)

    # Save down summary of URLs and saved processed files
    summary = {"parsed_pds": all_parsed_pdfs}
    summary_json = json.dumps(summary)
    with open(os.path.join(FILE_PATH, "summary.json"), 'w') as f: 
        f.write(summary_json)
