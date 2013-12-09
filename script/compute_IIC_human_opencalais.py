#!/usr/local/bin/env python2.7
# Name: Seungwon Yang  August. 5, 2013
# Filename: compute_IIC_human_opencalais.py
# Description: this script computes Rolling's Inter Indexer Consistency
#              given topic groups from human indexers as well as from
#              OpenCalais NLP API.
# Usage: #>python compute_IIC_human_opencalais.py
# Note: db access information is hard-coded, so change it for your purpose.

import sys
import re
import codecs
import json
import logging
import MySQLdb
import requests
import string
import urllib2
]from math import log, sqrt
from nltk import pos_tag, word_tokenize, stem
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import *
from operator import itemgetter # to sort dictionaries by their keys

class CosSim:
	# make a stopword set
	def __init__(self, stop_path, custom_stop_path):
		stop_li = open(stop_path, "r").read().split()
		custom_stop_li = open(custom_stop_path, "r").read().split()
		self.stoplist = set(stop_li + custom_stop_li)
		self.wdict = {}
		self.sum_dict = {}
		self.dcount = 0	

	def removeSymLemmatize(self, string_space_sep):
		lemm = WordNetLemmatizer()
		no_symbol = re.sub(r'[^\w]', ' ', string_space_sep)
		parsedLi = no_symbol.split()
		cleanLi = []
		for w in parsedLi:
			if (w in self.stoplist) or (len(w) < 3):
				continue
			else:
				cleanLi.append(lemm.lemmatize(w))
		return cleanLi

	def computeRolling(self, a_li, b_li):
		rolling = 0.0
		numA = len(a_li)
		numB = len(b_li)
		intersect_li = list(set(a_li) & set(b_li))
		# union_li = list(set(a_li) | set(b_li))
		numIntersection = len(intersect_li)
		# numUnion = len(union_li)
		rolling = float(2*numIntersection)/float(numA + numB)
		return rolling

	def extData(self, host, user, passwd, db, dbtable, dbtable2, human_id_li):
		db = MySQLdb.connect(host, user, passwd, db) 
		cur = db.cursor()
		micro_corpus = []
		# loop through 30 documents
		for i in range(1,31):
			corpus_per_topics = []
			# get the topics from multiple human indexers
			for j in human_id_li:
			  query = "select topics" + str(i) + " from "+dbtable+" where participant_id=" + str(j)
			  cur.execute(query)
			  single_doc_in_corpus = " ".join(cur.fetchone()[0].split(","))
			  corpus_per_topics.append(single_doc_in_corpus)
			# for multiple human indexers, get topics from OpenCalais only one-time
			query2 = "select topic from "+dbtable2+" where doc_id=" + str(i)
			cur.execute(query2)
			opencalais_doc_in_corpus = " ".join(cur.fetchone()[0].split(","))
			corpus_per_topics.append(opencalais_doc_in_corpus)
			micro_corpus.append(corpus_per_topics)
		return micro_corpus

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	host = "your.host.name.here"
	# ------------ DB access info ----------- #
	user = "db username"
	passwd = "db password"
	db = "db name"
	dbtable = "human topic db table name"
	dbtable2 = "opencalais topic db table name"
	human_id_li = [11,12,13,14,15]  # the IDs of multiple human indexers
	cs = CosSim("stopwords.txt", "custom_stops.txt")
	extracted_data = cs.extData(host, user, passwd, db, dbtable, dbtable2, human_id_li)	
	keepNum = 1
	for i in range(0, len(human_id_li)):
		for j in range(i+1, len(human_id_li)+1): # include M topics
			rollingSum = 0
			rollingLi = []
			rollingDiff = []
			print "(%d, %d)" % (i+1,j+1)
			for doc in extracted_data:
				setA = cs.removeSymLemmatize(doc[i])
				setB = cs.removeSymLemmatize(doc[j])				
				rollingScore = cs.computeRolling(setA, setB)
				rollingLi.append(rollingScore)
			rollingSum = sum(rollingLi)
			# for an average, divided by 30 for 30 text document. 
			# change the value for your purpose
			rollingAve = float(rollingSum)/float(30)  
			rollingDiffSquare = [(roll-rollingAve)*(roll-rollingAve) for roll in rollingLi]
			rollingStddev = sqrt(sum(rollingDiffSquare)/float(len(rollingLi)))
			print "Rolling average: %0.3f	stddev:%0.3f -------\n" % (rollingAve, rollingStddev)
		keepNum += 1

if __name__ == "__main__":
	main()