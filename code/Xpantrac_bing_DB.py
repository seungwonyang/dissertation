# Name: Seungwon Yang  July 28, 2013
# Filename: Xpantrac_bing_DB.py
# Description: this script extracts api call results from api_results_50 DB table,
#              which was contructed by expanding  computes
#              20 topic tags for each document, and add them to nyt_1000 table for later 
#              P, R, F1 analysis.
#
# Usage: #>python Xpantrac_bing_DB.py
#
# ---------- Note: num_topics = 20 -----------
#
# Parameters:
#   * unit_size = [20,25,10,5]  (maybe include '1' as well)
#	* num_api_return = [50,10,1]	
#   * tf_idf = [1, 0]
#   * only_noun = [1, 0]

import ast  # to convert a list (in string type) to an actual list type
import sys
import re
import codecs
import httplib2
import json
import logging
import math
import MySQLdb
import oauth2
import requests
import simplejson
import string
import time
import urllib2

# ------- for a trained raubt tagger ---------------
import itertools, nltk.tag
from nltk.corpus import brown, conll2000, treebank
import pos_tagger
# --------------------------------------------------

# from gensim import corpora, models, similarities
from collections import Counter
from HTMLParser import HTMLParser
from math import log, sqrt
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import *
from operator import itemgetter # to sort dictionaries by their keys

# functions to compute intersection and union of topics
def computeIntersection(a_li, b_li):
	return list(set(a_li) & set(b_li))

def computeUnion(a_li, b_li):
	# return list(set(a_li) & set(b_li))
	return list(set(a_li) | set(b_li))

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
	
class Xpantrac:
	# make a stopword set
	def __init__(self, stop_path, custom_stop_path):
		stop_li = open(stop_path, "r").read().split()
		custom_stop_li = open(custom_stop_path, "r").read().split()
		self.stoplist = list(set(stop_li + custom_stop_li))
		self.wdict = {}
		self.sum_dict = {}
		self.dcount = 0	

	# this function removes symbol characters, and ignores foreign languages
	def removeSymbols(self, any_text):	
		# first, decode already encoded text
		utf8_str = any_text.decode('utf-8')
		# then, encode with ascii only
		only_ascii = utf8_str.encode('ascii', 'ignore')
		# only_ascii = any_text.encode('ascii', 'ignore')
		# remove symbol characters
		return re.sub(r'[^\w]', ' ', only_ascii)		
	
	def parseInput(self, text):
		clean_text = self.removeSymbols(text).lower()
		# use gensim.utils.simple_preprocess(doc) -------------------
		return [t for t in clean_text.split() if (len(t) > 1) and (t not in self.stoplist)]

	def parse2Dict(self, micro_corpus):
		# words = " ".join(micro_corpus).split();
		lm = WordNetLemmatizer()
		doc_list = []
		nounsOnly = ["NN", "NNS", "NNP"]
		for doc in micro_corpus:
			# check nounsOnly value
			pos_tag_li = pos_tag(word_tokenize(doc))
			for key,val in pos_tag_li:
				if val in nounsOnly:
					# doc_list.append(key.lower())
					doc_no_symbol = re.sub(r'[^\w]', '', key)
					doc_list.append(doc_no_symbol.lower())

			for w in doc_list:
				w = lm.lemmatize(w)
				if (w in self.stoplist) or (len(w) < 3):
					continue
				elif w in self.wdict:
					self.wdict[w].append(self.dcount)
				else:
					self.wdict[w] = [self.dcount]
			self.dcount += 1   

	def makeQueryUnits(self, parsed_input_li, unit_size=5, window_overlap=1):
	    unit = []
	    query_list = []
	    list_len = len(parsed_input_li)
	    # if unit_size ==1, no windowing.
	    if unit_size==1:
	    	for item in parsed_input_li:
	    		unit = [item]
	    		query_list.append(unit)
	    	return query_list
	    i = 0
	    while (i < list_len):
	    	unit = parsed_input_li[i:i+unit_size]
	    	if len(unit) < unit_size:  # last unit
	    		query_list.append(unit)
	    		return query_list
	    	else:
	    		query_list.append(unit)
	    		i += unit_size - window_overlap

	def makeYahooCorpus(self, query_list, num_api_return, yahoo_api_type):
		# -------------- how about using only the nouns (and compound nouns) ?
		# -------------- maybe use POS tagger?  which one?  
		# -------------- also consider lemmatization: gensim.utils.lemmatize(content)
	    OAUTH_CONSUMER_KEY = "yahoo api consumer key"
	    OAUTH_CONSUMER_SECRET = "yahoo api consumer secret"
	    micro_corpus = []
	    micro_corpus_0 = []
	    micro_corpus_1 = []
	    num_results_returned_li = []
  		# ------------ adjust query_list size here ---------------
	    # for item in query_list[:5]: # [[query unit 1], [query unit 2], [query unit 3],...]
	    for item in query_list: # [[query unit 1], [query unit 2], [query unit 3],...]
	        query = " ".join(item).replace(" ", "%20")
	        print query
	        num_results_returned = 0
	        if query != "":
	            try:
	                url = ""
	                if yahoo_api_type == "web":
	                    url = "http://yboss.yahooapis.com/ysearch/web?q=" + query
	                else:
	                    url = "http://yboss.yahooapis.com/ysearch/news?q=" + query
	                consumer = oauth2.Consumer(key=OAUTH_CONSUMER_KEY,secret=OAUTH_CONSUMER_SECRET)
	                params = {
	                    'oauth_version': '1.0',
	                    'oauth_nonce': oauth2.generate_nonce(),
	                    'oauth_timestamp': int(time.time()),
	                }
	                oauth_request = oauth2.Request(method='GET', url=url, parameters=params)
	                oauth_request.sign_request(oauth2.SignatureMethod_HMAC_SHA1(), consumer, None)
	                oauth_header=oauth_request.to_header(realm='yahooapis.com')
				    # Get search results
	                http = httplib2.Http()
	                resp, content = http.request(url, 'GET', headers=oauth_header)
	                results = simplejson.loads(content)
	                # combine all returned results into a single string and append it
	                group_str_0 = ""
	                group_str_1 = ""					
	                results_li = []
	                if yahoo_api_type == 'web':
						# for M_39 configuration (50 results merged) ------//
		                results_li = results['bossresponse']['web']['results']
	                else:
		                results_li = results['bossresponse']['news']['results']
	                # keep track of the number of results for each API query
	                num_results_returned = len(results_li)
	                # for result in results_li[0:20]:
	                for result in results_li:
	                    clean_result = strip_tags(result['abstract']).replace("...", "").strip().replace("\"", "")
	                    group_str_0 = group_str_0 + " " + clean_result
	                micro_corpus_0.append(group_str_0)
	                # for M_43 configuration (only max. 10 results merged) ------------//
	                for result in results_li[0:10]:
	                    clean_result = strip_tags(result['abstract']).replace("...", "").strip().replace("\"", "")
	                    group_str_1 = group_str_1 + " " + clean_result
	                micro_corpus_1.append(group_str_1)

	            except Exception, err:
	                #sys.stderr.write('ERROR: %s\n' % str(err))
	                print "Something wrong............"
	                pass
	        else:
	        	num_results_returned = 0
	        num_results_returned_li.append(num_results_returned)

	    micro_corpus.append(micro_corpus_0)
	    micro_corpus.append(micro_corpus_1)
	    micro_corpus.append(num_results_returned_li)

	    print micro_corpus[0]
	    print "\n--------------------------------\n"
	    print micro_corpus[1]
	    print "\n--------------------------------\n"
	    print micro_corpus[2]
	    print "\n--------------------------------\n"
	    # print "Size of the micro_corpus: %d" % len(micro_corpus)
	    # micro_corpus = [corpus_50, corpus_10, api_return_count_li]
	    return micro_corpus

	def makeCorpusFromDB(self, db_cursor, doc_id, dbtable):
	    micro_corpus = []
	    micro_corpus_0 = []  # holds corpus with 50 returns (M39)
	    micro_corpus_1 = []  # holds corpus with 10 returns (M43)
	    # construct db query to collect all query results of doc_id
	    query = "select query_id,query,results from " + dbtable + " where doc_id=" + str(doc_id)
	    db_cursor.execute(query)
	    for item in db_cursor.fetchall():
	    	# print item[0]
	    	# print item[1]
	    	result_str = item[2]	    	
	    	# For M_39 (50 result) - use all result_li
	    	result_li = ast.literal_eval(result_str) # to convert string to a python list type
	    	micro_corpus_0.append(" ".join(result_li))
	    	# For M_43 (10 result) - use top 10 resulting descriptions
	    	micro_corpus_1.append(" ".join(result_li[0:10]))
	    return [micro_corpus_0, micro_corpus_1]

	def makeCorpus_progressive(self, db_cursor, doc_id, dbtable):
	    micro_corpus = []
	    micro_corpus_0 = []
	    micro_corpus_1 = []
	    # construct db query to collect all query results of doc_id
	    query = "select query_id,query,results from " + dbtable + " where doc_id=" + str(doc_id)
	    db_cursor.execute(query)
	    # compute number of total queries
	    returned_results_li = db_cursor.fetchall()
	    num_queries = len(returned_results_li)
	    for item in returned_results_li:
	    	# print item[0]
	    	# print item[1]
	    	result_str = item[2]
	    	
	    	# For M_39 (50 result) - use all result_li
	    	result_li = ast.literal_eval(result_str)
	    	micro_corpus_0.append(" ".join(result_li))

	    	# For M_43 (10 result) - use top 10 resulting descriptions
	    	micro_corpus_1.append(" ".join(result_li[0:10]))
	    half_num_queries = num_queries / 2
	    return [micro_corpus_0[0:half_num_queries],micro_corpus_0[half_num_queries:-1], micro_corpus_1[0:half_num_queries],micro_corpus_1[half_num_queries:-1]]

	def makeCorpus_weighted(self, db_cursor, doc_id, dbtable, weight_ratio):
		# weight_ratio: (e.g., 0.2 for weighting top 20% of input text)
	    micro_corpus = []
	    micro_corpus_0 = []
	    micro_corpus_1 = []

	    # construct db query to collect all query results of doc_id
	    query = "select query_id,query,results from " + dbtable + " where doc_id=" + str(doc_id)
	    db_cursor.execute(query)
	    # compute number of total queries
	    returned_results_li = db_cursor.fetchall()
	    num_queries = len(returned_results_li)
	    # compute duplication iteration
	    dup_iter = int(math.ceil(num_queries * weight_ratio))
	    print "dup_iter: %d ---------" % dup_iter

	    for item in returned_results_li:
	    	# print item[0]
	    	# print item[1]
	    	result_str = item[2]
	    	
	    	# For M_39 (50 result) - use all result_li
	    	result_li = ast.literal_eval(result_str)
	    	micro_corpus_0.append(" ".join(result_li))

	    	# For M_43 (10 result) - use top 10 resulting descriptions
	    	micro_corpus_1.append(" ".join(result_li[0:10]))

	    # weight the corpus with duplicate data -----------------
	    for item in returned_results_li[0:dup_iter]:
	    	result_str = item[2]
	    	
	    	# For M_39 (50 result) - use all result_li
	    	result_li = ast.literal_eval(result_str)
	    	micro_corpus_0.append(" ".join(result_li))

	    	# For M_43 (10 result) - use top 10 resulting descriptions
	    	micro_corpus_1.append(" ".join(result_li[0:10]))
	    return [micro_corpus_0, micro_corpus_1]

	def build(self):
		# collect keys which appear more than once in the corpus
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys.sort()
		num_keys = len(self.keys)
		min_cell_val = 1 / float(num_keys)
		# create a zero matrix of terms (i.e., keys) and document numbers (i.e., columns)
		self.A = zeros([len(self.keys), self.dcount])
		# update cell values with wdict values
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:               
				self.A[i,d] = self.A[i,d] + 1
		
		# ------ if a column sum is 0, fill it with min_cell_val
		WordsPerDoc = sum(self.A, axis=0)
		zero_col_index = where(WordsPerDoc==0)
		for index in zero_col_index:
			self.A[:,index] = min_cell_val

	# this function adds all cell values for each key (i.e., row)
	def build_sum_dict(self):
	    rows, cols = self.A.shape 
	    for i, k in enumerate(self.keys):
	        row_sum = 0
	        for j in range(cols):
	            row_sum = row_sum + self.A[i, j]        
	        self.sum_dict[k] = row_sum     
	        
	def extTopics(self, num_topics):		
	    # sort self.sum_dict in descending order
		sorted_list = sorted(self.sum_dict.iteritems(), key=itemgetter(1), reverse=True)
	    # select top num_topics and print them out
		topics_li = []
		topics_str = ""
		for item, score in sorted_list[:num_topics]:
			topics_li.append(item)
		topics_str = ",".join(topics_li)
		return topics_str

	# extract topics by using Counter() given a micro_corpus
	def extTopicsCounter(self, micro_corpus, num_topics, trained_tagger):
		# words = " ".join(micro_corpus).split();
		lm = WordNetLemmatizer()
		key_list = []
		count_list = []
		nounsOnly = ["NN", "NNS", "NNP"]
		for doc in micro_corpus:
			# check nounsOnly value

			# pos_tag_li = pos_tag(word_tokenize(doc))
			pos_tag_li = trained_tagger.tag(word_tokenize(doc))

			for key,val in pos_tag_li:
				if val in nounsOnly:
					# doc_list.append(key.lower())
					key_no_symbol = re.sub(r'[^\w]', '', key)
					
					w = key_no_symbol.lower()
					if (w not in self.stoplist) or (len(w) > 2):

						w = lm.lemmatize(w)
						
						count_list.append(w)
		topic_list = []
		# count the highest frequency N words in count_list
		for word in Counter(count_list).most_common(num_topics):
			topic_list.append(word[0])
		topic_str = ",".join(topic_list)

		return topic_str

	# def printTopics2Db(self, num_topics, dbuser, dbpasswd, hostname, dbname, dbtable, iter_id, unit_size, num_api_return, tfidf, only_nouns, input_doc_id):
	def printTopics2Db(self, num_topics, dbuser, dbpasswd, hostname, dbname, dbtable, iter_id, unit_size, num_api_return, df, only_nouns, input_doc_id):
	    # sort self.sum_dict in descending order
		sorted_list = sorted(self.sum_dict.iteritems(), key=itemgetter(1), reverse=True)
	    # select top num_topics and print them out
		topics_li = []
		topics_str = ""
		for item, score in sorted_list[:num_topics]:
			topics_li.append(item)
		topics_str = ",".join(topics_li)
		print topics_str
		# connect to mysql
		db = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
		cur = db.cursor() 
		query = "update "+ dbtable +" set unit_size='" +str(unit_size)+ "', num_api_return='" + str(num_api_return) + "', df='" + str(df) + "', only_nouns='" + str(only_nouns) + "', topics" + str(input_doc_id) +"='" + topics_str + "' where id=" + str(iter_id)
		cur.execute(query)		

	def calc_cosine_sim(self, u, v):
		return dot(u, v) / (sqrt(dot(u,u)) * sqrt(dot(v,v)))

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# parameters ------------------------------------ (Overall)
	num_topics = 20
	window_overlap = 1
	# ----------------------------------------------- (nyt_1000)
	dbuser = "db username"
	dbpasswd = "db password"
	hostname = "db hostname"
	dbname = "nyt"
	dbtable = "api_results_50_bing"
	# dbtable2 = "nyt_3000"
	dbtable2 = "nyt_1000"
	# ----------------------------- Train custom POS tagger ------------
	pt = pos_tagger.PosTagger(brown, conll2000, treebank)	
	# train using the ENTIRE conll2000 corpus
	raubt_trained = pt.train_raubt_tagger(conll2000.tagged_sents())
	# ------------------------------------------------------------------

	# develop id list
	fi = open("NYT_1000_IDs_EXP3.txt", "r")
	li = fi.read().split()
	fi.close()

    # connect to mysql
	db = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
	cur = db.cursor()
	# input text ------------------------------------- (Control inputs)
	micro_corpus = []
	iter_id = 1
	for filenum in li[0:1000]:  # doc id 1-1000
		# compute processing time for each file
		start_time = time.time()
		print start_time
		text = ""
		m_39_topics = ""
		m_43_topics = ""
		topic_inter_str = ""
		topic_uni_str = ""
		doc_id = filenum

		print "\n------------- Document ID: %s is being processed --------------\n" % doc_id

		xpt_0 = Xpantrac("stopwords.txt", "custom_stops.txt")
		xpt_1 = Xpantrac("stopwords.txt", "custom_stops.txt")

		# ------ Extract topic tags from DB table 'api_results_50' -----------------		
		micro_corpus = xpt_0.makeCorpusFromDB(cur, doc_id, dbtable)

		# -------------------- Frequency-Based ------------------------------- #		
		# process micro_corpus ONLY when it is not empty -----//
		if micro_corpus[0] != []:						
			print "------- m39_20 ---------"
			m_39_topics = xpt_0.extTopicsCounter(micro_corpus[0], num_topics, raubt_trained)
			print m_39_topics

			print "------- m43_20 ---------"
			m_43_topics = xpt_1.extTopicsCounter(micro_corpus[1], num_topics, raubt_trained)
			print m_43_topics
			print "\n"
			
			m_39_li = m_39_topics.split(',')
			m_43_li = m_43_topics.split(',')

			topic_intersection = computeIntersection(m_39_li, m_43_li)
			topic_inter_str = ",".join(topic_intersection)
			topic_union = computeUnion(m_39_li, m_43_li)
			topic_uni_str = ",".join(topic_union)

		# -------- Write to DB table, 'nyt_1000'
		query4 = "update "+ dbtable2 +" set M39_bing='" + m_39_topics + "', M43_bing='" + m_43_topics + "', 39_AND_43_bing='" + topic_inter_str + "', 39_OR_43_bing='" + topic_uni_str + "' where doc_id=" + str(doc_id)
		cur.execute(query4)
		end_time = time.time()
		print end_time - start_time, "seconds"

		# --------------------------- (Update iteration)
        iter_id += 1				

if __name__ == "__main__":
	main()