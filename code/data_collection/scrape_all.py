import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

HOME_URL = "https://aclanthology.org/"
acl_page = requests.get(HOME_URL)
soup = BeautifulSoup(acl_page.content, "html.parser")

sources = []
for a in soup.find_all('a', href=True):
  if 'events' in a['href']:
    sources.append(HOME_URL+a['href'][1:])

pdf_links = []

print('Collecting PDF URLs from {} conferences!'.format(len(sources)))
for url in tqdm(sources, total=len(sources)):
  page = requests.get(url)

  soup = BeautifulSoup(page.content, "html.parser")

  for a in soup.find_all('a', title="Open PDF"):
    pdf_links.append(a['href'])

print('Successfully collected {} PDF URLs!'.format(len(pdf_links)))

# write to file
out_file = "pdf_urls.txt"
print('Writing PDF URLs to {}!'.format(out_file))
with open(out_file, 'w') as f:
    for line in tqdm(pdf_links, total=len(pdf_links)):
        f.write(line+'\n')

