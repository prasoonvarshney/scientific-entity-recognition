import requests
import PyPDF2
import pandas as pd

from tqdm import tqdm
from io import BytesIO
from bs4 import BeautifulSoup


EMNLP_URL = "https://aclanthology.org/volumes/2021.emnlp-main/"
ACL_URL = "https://aclanthology.org/volumes/2022.acl-long/"


# Scrape ACL Anthology
sources = [ACL_URL, EMNLP_URL]
links = []

for url in sources:
  page = requests.get(url)

  soup = BeautifulSoup(page.content, "html.parser")

  for a in soup.find_all('a', title="Open PDF"):
    links.append(a['href'])


# Read PDFs from URLs 
full_text = []

for url in tqdm(links):
  response = requests.get(url)
  my_raw_data = response.content

  text = []

  with BytesIO(my_raw_data) as data:
      read_pdf = PyPDF2.PdfFileReader(data)

      for page in range(read_pdf.getNumPages()):
          text.append((read_pdf.getPage(page).extractText()).replace('\n', ' '))

  full_text.append(text[0])


# Save as CSV
df = pd.DataFrame()
df['link'] = links[:5]
df['full_text'] = full_text
df.to_csv('nlp_papers.csv')