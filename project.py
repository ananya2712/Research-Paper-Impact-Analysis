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
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')
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

def main():

	# st.markdown('<style>body{background-color: MidnightBlue;}</style>',unsafe_allow_html=True)
	
	# create instance of semantic scholar class
	sch = SemanticScholar(timeout=2)
	# Sidebar contents 
	menu = ["Home","Overview","Visualize"]
	
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		# st.subheader("Home")
		st.markdown(html_temp.format('indigo','white'),unsafe_allow_html=True)
		st.markdown("<h1 style='text-align: center; color: AliceBlue; font-size:25px'>A simple and interactive tool to visualise research paper impact</h1>", unsafe_allow_html=True)

	if choice == "Overview":
		doi = get_doi()
		paper = sch.paper(doi)
		for key in paper.keys():
			if key not in ["citations","references","isOpenAccess","isPublisherLicensed","is_publisher_licensed","is_open_access","topics","venue"]:
				st.write(f"{key} : {paper[key]}") 

	if choice == "Visualize":
		doi = get_doi()
		paper = sch.paper(doi)
		st.text("In Citations vs Out Citations")
		bar_chart_in_out(paper)
		st.text("Citations Grouped by Years")
		cpi(paper)



if __name__ == '__main__':
	main()