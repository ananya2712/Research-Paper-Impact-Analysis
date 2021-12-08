# LIBRARIES

# BIOLERPLATE
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import string
from collections import Counter, defaultdict
from enum import Enum
from itertools import chain, groupby, product
# NLP
import nltk
from nltk import tokenize
from operator import itemgetter
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import re
from rake_nltk import Rake
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')
from typing import Callable, DefaultDict, Dict, List, Optional, Set, Tuple
# Front End
import streamlit as st 

# API Requests
from typing_extensions import Literal
import requests

# TEMPLATES 
html_temp = """
<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:{};text-align:center;">RePI</h1>
</div>
"""

# SEMANTIC SCHOLAR API REQUESTS CLASS
from typing_extensions import Literal
import requests


class SemanticScholar:

    DEFAULT_API_URL = 'https://api.semanticscholar.org/v1'
    DEFAULT_PARTNER_API_URL = 'https://partner.semanticscholar.org/v1'

    auth_header = {}

    def __init__(self,timeout: int=2,api_key: str=None,api_url: str=None) -> None:

        if api_url:
            self.api_url = api_url
        else:
            self.api_url = self.DEFAULT_API_URL

        if api_key:
            self.auth_header = {'x-api-key': api_key}
            if not api_url:
                self.api_url = self.DEFAULT_PARTNER_API_URL

        self.timeout = timeout

    def paper(self, id: str, include_unknown_refs: bool=False) -> dict:
        # Paper lookup
        # param str id: S2PaperId, DOI or ArXivId.
        data = self.get_data('paper', id, include_unknown_refs)
        return data

    def author(self, id: str) -> dict:
        data = self.get_data('author', id, False)
        return data

    def get_data(self,method: Literal['paper', 'author'],id: str ,include_unknown_refs: bool) -> dict:

        data = {}
        method_types = ['paper', 'author']
        if method not in method_types:
            raise ValueError(
                'Invalid method type. Expected one of: {}'.format(method_types)
            )

        url = '{}/{}/{}'.format(self.api_url, method, id)
        if include_unknown_refs:
            url += '?include_unknown_references=true'
        r = requests.get(url, timeout=self.timeout, headers=self.auth_header)

        if r.status_code == 200:
            data = r.json()
            if len(data) == 1 and 'error' in data:
                data = {}
        elif r.status_code == 403:
            raise PermissionError('HTTP status 403 Forbidden.')
        elif r.status_code == 429:
            raise ConnectionRefusedError('HTTP status 429 Too Many Requests.')

        return data

def get_doi():
	doi = st.text_input('Enter DOI here')
	return doi

# Bar charts for in and out citations
def bar_chart_in_out(paper) -> None :
  dat = [paper['numCitedBy'], paper['numCiting']]
  dat = [('Cited by',paper['numCitedBy']), ('Citing',paper['numCiting'])]
  labels, ys = zip(*dat)
  ys = np.asarray(ys)
  l = np.asarray(labels)
  fig = plt.figure(figsize=(10, 7))
  bx = sns.barplot(x=l ,y= ys, palette="Blues_d")
  st.pyplot(fig)

# Group Citations by Year
def cpi(paper) -> None :
  cits = paper['citations']
  years = []
  for i in cits:
    years.append(i.get('year'))
  years = np.array(years)
  (unique, counts) = np.unique(years, return_counts=True)
  sns.set(rc = {'figure.figsize':(15,8)})
  fig = plt.figure(figsize=(10,7))
  cx = sns.barplot(x = unique, y = counts)
  cx.set(xlabel = 'years', ylabel = 'number of citing papers')  
  st.pyplot(fig)


def pre_process(abstract):
	stop_words = set(stopwords.words('english'))
	abstract = re.sub(r'[^\w\s]','',abstract)
	word_tokens = word_tokenize(abstract) # tokenise the words
	words = [w for w in word_tokens if not w.lower() in stop_words] #filter the words
	return words

# Function to check if the word is present in a sentence list
def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

# Return top n-most keywords
def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

 # TF-IDF for Keyword Extraction
def tf_idf(abstract):
	words = pre_process(abstract)
	stop_words = set(stopwords.words('english'))

	# Total Words in the Abstract
	total_words = abstract.split()
	total_word_length = len(total_words)

	# Total Sentences in the Abstract
	total_sentences = tokenize.sent_tokenize(abstract)
	total_sent_len = len(total_sentences)

	# Calculate TF - Term Frequency 
	tf_score = {}
	for each_word in words:
		each_word = each_word.replace('.','')
		if each_word not in stop_words:
			if each_word in tf_score:
				tf_score[each_word] += 1
			else:
				tf_score[each_word] = 1

	tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())

	# Calculate IDF - Inverse Document Frequency 
	idf_score = {}
	for each_word in words:
		each_word = each_word.replace('.','')
		if each_word not in stop_words:
			if each_word in idf_score:
				idf_score[each_word] = check_sent(each_word, total_sentences)
			else:
				idf_score[each_word] = 1

	idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

	tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
	res = get_top_n(tf_idf_score, 15)
	return res

# Generate Word Cloud	
def wrdcld(keywords)->None: 
  unique_string=(" ").join(keywords)
  wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
  fig = plt.figure(figsize=(20,8))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.show()
  st.pyplot(fig)

def main():

	# st.markdown('<style>body{background-color: MidnightBlue;}</style>',unsafe_allow_html=True)
	
	# create instance of semantic scholar class
	sch = SemanticScholar(timeout=2)
	# Sidebar contents 
	menu = ["Home","Overview","Visualize","Impact Factor"]
	
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		# st.subheader("Home")
		st.markdown(html_temp.format('indigo','white'),unsafe_allow_html=True)
		st.markdown("<h1 style='text-align: center; color: AliceBlue; font-size:25px'>A simple and interactive tool to visualise research paper impact</h1>", unsafe_allow_html=True)

	if choice == "Overview":
		doi = get_doi()
		k = []
		v = []
		if len(doi) != 0 :
			paper = sch.paper(doi)
			for key in paper.keys():
				if key not in ["citations","references","isOpenAccess","isPublisherLicensed","is_publisher_licensed","is_open_access","topics","venue"]:
					#st.write(f"{key} : {paper[key]}")
					k.append(key)
					v.append(str(paper[key]))
			df = pd.DataFrame(list(zip(k, v)),columns =['Key', 'Value'])
			st.table(df)
			
					

	if choice == "Visualize":
		doi = get_doi()
		if len(doi) != 0:
			paper = sch.paper(doi)

			# GRAPHS 
			st.subheader("In Citations vs Out Citations")
			bar_chart_in_out(paper)
			st.subheader("Citations Grouped by Years")
			cpi(paper)

			# KEYWORD EXTRACTION 
			abstract = paper['abstract']

			tf_idf_result = tf_idf(abstract) # returns keywords 
			keywords1 = tf_idf_result.keys()
			keywords1 = list(keywords1)
			# st.write(keywords)
			# wrdcld(keywords)
			st.subheader("Wordcloud of keywords for the paper")
			r = Rake()
			r.extract_keywords_from_text(abstract)
			keywords = r.get_ranked_phrases()
			#st.write(f"{keywords}")
			wrdcld(keywords)

			#keywords wordcloud for citing papers
			st.subheader("Wordcloud of keywords for all the works citing the paper")
			cits = paper['citations']
			dois = []
			for i in cits:
				dois.append(i.get('doi'))
			res = [i for i in dois if i]
			keyw1 = []
			count = 0
			for d in dois:
				if count == 30:
					break
				pap = sch.paper(res[1]) 
				r.extract_keywords_from_text(pap['abstract'])
				keyw2 = r.get_ranked_phrases()
				keyw1.extend(keyw2)
				count +=1
			
			#print(keyw1)
			wrdcld(keyw1)


			

	if choice == "Impact Factor":
		doi = get_doi()
		if len(doi) != 0 :
			paper = sch.paper(doi)
			im = 0 
			im = paper['influentialCitationCount']/paper['numCitedBy']
			st.write(f"Impact Factor : {im}")

if __name__ == '__main__':
	main()