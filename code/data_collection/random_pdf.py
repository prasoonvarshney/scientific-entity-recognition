import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=str, default="5")
args = parser.parse_args()
n = int(args.n)

FILENAME = 'pdf_urls.txt'

all_files = []

with open(FILENAME, 'rb') as f:
    for line in f.readlines():
        URL = str(line)[:-3]
        file = str(line)[:-3].split('/')[-1] + '.txt'
        all_files.append((file, URL))


PARSED_PDFS_DIR = 'parsed_pdfs'
parsed_pdfs = os.listdir(PARSED_PDFS_DIR)

pool = []

for file, url in all_files:
    if file not in parsed_pdfs:
        pool.append(url)

chosen = np.random.choice(pool, n, replace=False)

for url in chosen:
    print(url[2:])