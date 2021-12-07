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
from wordcloud import WordCloud
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

def main():

	# st.markdown('<style>body{background-color: MidnightBlue;}</style>',unsafe_allow_html=True)
	
	menu = ["Home","Search","Visualize"]
	
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		# st.subheader("Home")
		st.markdown(html_temp.format('indigo','white'),unsafe_allow_html=True)
		st.markdown("<h1 style='text-align: center; color: AliceBlue; font-size:25px'>A simple and interactive tool to visualise research paper impact</h1>", unsafe_allow_html=True)

	if choice == "Search":
		doi = st.text_input('Enter DOI here')	
		

if __name__ == '__main__':
	main()