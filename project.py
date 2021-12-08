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
#from rake_nltk import Rake
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

#rake from scratch trials
# Readability type definitions.
Word = str
Sentence = str
Phrase = Tuple[str, ...]


class Metric(Enum):
	"""Different metrics that can be used for ranking."""

	DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
	WORD_DEGREE = 1  # Uses d(w) alone as the metric
	WORD_FREQUENCY = 2  # Uses f(w) alone as the metric



class Rake:
	"""Rapid Automatic Keyword Extraction Algorithm."""

	def __init__(
		self,
		stopwords: Optional[Set[str]] = None,
		punctuations: Optional[Set[str]] = None,
		language: str = 'english',
		ranking_metric: Metric = Metric.DEGREE_TO_FREQUENCY_RATIO,
		max_length: int = 100000,
		min_length: int = 1,
		include_repeated_phrases: bool = True,
		sentence_tokenizer: Optional[Callable[[str], List[str]]] = None,
		word_tokenizer: Optional[Callable[[str], List[str]]] = None,
	):

		# By default use degree to frequency ratio as the metric.
		if isinstance(ranking_metric, Metric):
			self.metric = ranking_metric
		else:
			self.metric = Metric.DEGREE_TO_FREQUENCY_RATIO

		# If stopwords not provided we use language stopwords by default.
		self.stopwords: Set[str]
		if stopwords:
			self.stopwords = stopwords
		else:
			self.stopwords = set(nltk.corpus.stopwords.words(language))

		# If punctuations are not provided we ignore all punctuation symbols.
		self.punctuations: Set[str]
		if punctuations:
			self.punctuations = punctuations
		else:
			self.punctuations = set(string.punctuation)

		# All things which act as sentence breaks during keyword extraction.
		self.to_ignore: Set[str] = set(chain(self.stopwords, self.punctuations))

		# Assign min or max length to the attributes
		self.min_length: int = min_length
		self.max_length: int = max_length

		# Whether we should include repeated phreases in the computation or not.
		self.include_repeated_phrases: bool = include_repeated_phrases

		# Tokenizers.
		self.sentence_tokenizer: Callable[[str], List[str]]
		if sentence_tokenizer:
			self.sentence_tokenizer = sentence_tokenizer
		else:
			self.sentence_tokenizer = nltk.tokenize.sent_tokenize
		self.word_tokenizer: Callable[[str], List[str]]
		if word_tokenizer:
			self.word_tokenizer = word_tokenizer
		else:
			self.word_tokenizer = nltk.tokenize.wordpunct_tokenize

		# Stuff to be extracted from the provided text.
		self.frequency_dist: Dict[Word, int]
		self.degree: Dict[Word, int]
		self.rank_list: List[Tuple[float, Sentence]]
		self.ranked_phrases: List[Sentence]

	def extract_keywords_from_text(self, text: str):
		sentences: List[Sentence] = self._tokenize_text_to_sentences(text)
		self.extract_keywords_from_sentences(sentences)

	def extract_keywords_from_sentences(self, sentences: List[Sentence]):
		phrase_list: List[Phrase] = self._generate_phrases(sentences)
		self._build_frequency_dist(phrase_list)
		self._build_word_co_occurance_graph(phrase_list)
		self._build_ranklist(phrase_list)

	def get_ranked_phrases(self) -> List[Sentence]:
		return self.ranked_phrases

	def _tokenize_text_to_sentences(self, text: str) -> List[Sentence]:
		"""Tokenizes the given text string into sentences using the configured
		sentence tokenizer. Configuration uses `nltk.tokenize.sent_tokenize`
		by default.
		:param text: String text to tokenize into sentences.
		:return: List of sentences as per the tokenizer used.
		"""
		return self.sentence_tokenizer(text)

	def _tokenize_sentence_to_words(self, sentence: Sentence) -> List[Word]:
		"""Tokenizes the given sentence string into words using the configured
		word tokenizer. Configuration uses `nltk.tokenize.wordpunct_tokenize`
		by default.
		:param sentence: String sentence to tokenize into words.
		:return: List of words as per the tokenizer used.
		"""
		return self.word_tokenizer(sentence)

	def _build_frequency_dist(self, phrase_list: List[Phrase]) -> None:
		self.frequency_dist = Counter(chain.from_iterable(phrase_list))

	def _build_word_co_occurance_graph(self, phrase_list: List[Phrase]) -> None:
		co_occurance_graph: DefaultDict[Word, DefaultDict[Word, int]] = defaultdict(lambda: defaultdict(lambda: 0))
		for phrase in phrase_list:
			# For each phrase in the phrase list, count co-occurances of the
			# word with other words in the phrase.
			#
			# Note: Keep the co-occurances graph as is, to help facilitate its
			# use in other creative ways if required later.
			for (word, coword) in product(phrase, phrase):
				co_occurance_graph[word][coword] += 1
		self.degree = defaultdict(lambda: 0)
		for key in co_occurance_graph:
			self.degree[key] = sum(co_occurance_graph[key].values())

	def _build_ranklist(self, phrase_list: List[Phrase]):
		"""Method to rank each contender phrase using the formula
			  phrase_score = sum of scores of words in the phrase.
			  word_score = d(w) or f(w) or d(w)/f(w) where d is degree
						   and f is frequency.
		:param phrase_list: List of List of strings where each sublist is a
							collection of words which form a contender phrase.
		"""
		self.rank_list = []
		for phrase in phrase_list:
			rank = 0.0
			for word in phrase:
				if self.metric == Metric.DEGREE_TO_FREQUENCY_RATIO:
					rank += 1.0 * self.degree[word] / self.frequency_dist[word]
				elif self.metric == Metric.WORD_DEGREE:
					rank += 1.0 * self.degree[word]
				else:
					rank += 1.0 * self.frequency_dist[word]
			self.rank_list.append((rank, ' '.join(phrase)))
		self.rank_list.sort(reverse=True)
		self.ranked_phrases = [ph[1] for ph in self.rank_list]

	def _generate_phrases(self, sentences: List[Sentence]) -> List[Phrase]:
		phrase_list: List[Phrase] = []
		# Create contender phrases from sentences.
		for sentence in sentences:
			word_list: List[Word] = [word.lower() for word in self._tokenize_sentence_to_words(sentence)]
			phrase_list.extend(self._get_phrase_list_from_words(word_list))

		# Based on user's choice to include or not include repeated phrases
		# we compute the phrase list and return it. If not including repeated
		# phrases, we only include the first occurance of the phrase and drop
		# the rest.
		if not self.include_repeated_phrases:
			unique_phrase_tracker: Set[Phrase] = set()
			non_repeated_phrase_list: List[Phrase] = []
			for phrase in phrase_list:
				if phrase not in unique_phrase_tracker:
					unique_phrase_tracker.add(phrase)
					non_repeated_phrase_list.append(phrase)
			return non_repeated_phrase_list

		return phrase_list

	def _get_phrase_list_from_words(self, word_list: List[Word]) -> List[Phrase]:
		groups = groupby(word_list, lambda x: x not in self.to_ignore)
		phrases: List[Phrase] = [tuple(group[1]) for group in groups if group[0]]
		return list(filter(lambda x: self.min_length <= len(x) <= self.max_length, phrases)) 

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
		st.markdown("<h4 style='text-align: center; color: AliceBlue; font-size:15px'>RePI  helps you visualise the Impact of any published Research Paper or Article based on the Digital Object Identifier (DOI)</h4>", unsafe_allow_html=True)



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

			wrdcld(keyw1)			

	if choice == "Impact Factor":
		doi = get_doi()
		if len(doi) != 0 :
			paper = sch.paper(doi)
			im = 0 
			im = paper['influentialCitationCount']/paper['numCitedBy']
			st.write("Impact Factor is a metric which gauges the paper impact.It can be calculated using the following formula :")
			st.markdown("<h3 style='text-align: center; color: white;'>I.M = influentialCitationCount / numCitedBy</h1>", unsafe_allow_html=True)
			st.write("According to some naive trials , an impact factor above 0.05 represents a good paper.")
			st.markdown(f"Impact Factor : {im}")

if __name__ == '__main__':
	main()