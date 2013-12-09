# Name: Seungwon Yang  May 23, 2013
# Filename: Xpantrac_extractTopics_bing.py
# Description: this script finds topics (i.e., significant words) of 
#              a text document using expansion-extraction of an input text.
#			   Found topics are stored in a database.
# Note: Bing search API is used.
# Usage: #>python Xpantrac_extractTopics_bing.py
#
# Parameters:
#   * unit_size = [20,25,10,5]  (maybe include '1' as well)
#	* num_api_return = [50,10,1]	
#   * tf_idf = [1, 0]
#   * only_noun = [1, 0]

import sys
import re
import codecs
import json
import logging
import MySQLdb
import requests
import string
import urllib2

# from gensim import corpora, models, similarities
from math import log, sqrt
from nltk import pos_tag, word_tokenize
from numpy import *
from operator import itemgetter # to sort dictionaries by their keys

# functions to compute intersection and union of topics
def computeIntersection(a_li, b_li):
	return list(set(a_li) & set(b_li))

def computeUnion(a_li, b_li):
	# return list(set(a_li) & set(b_li))
	return list(set(a_li) | set(b_li))

class Xpantrac:
	# make a stopword set
	def __init__(self, stop_path, custom_stop_path):
		stop_li = open(stop_path, "r").read().split()
		custom_stop_li = open(custom_stop_path, "r").read().split()
		self.stoplist = set(stop_li + custom_stop_li)
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

	def parse2Dict(self, micro_corpus, nounsOnly=[]):
		# words = " ".join(micro_corpus).split();
		doc_list = []
		for doc in micro_corpus:
			# check nounsOnly value
			if nounsOnly != []:
				pos_tag_li = pos_tag(word_tokenize(doc))
				for key,val in pos_tag_li:
					if val in nounsOnly:
						# doc_list.append(key.lower())
						doc_no_symbol = re.sub(r'[^\w]', '', key)
						doc_list.append(doc_no_symbol.lower())
			else:
				doc_no_symbol = re.sub(r'[^\w]', ' ', doc)
				doc_list = doc_no_symbol.lower().split() 
			for w in doc_list:
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

	def makeMicroCorpus(self, query_list, num_api_return):
		# -------------- how about using only the nouns (and compound nouns) ?
		# -------------- maybe use POS tagger?  which one?  
		# -------------- also consider lemmatization: gensim.utils.lemmatize(content)
	    micro_corpus = []
	    micro_corpus_0 = []
	    micro_corpus_1 = []
	    num_results_returned_li = []
        # ------------ adjust query_list size here ---------------
	    for item in query_list: # [[query unit 1], [query unit 2], [query unit 3],...]
	        query = " ".join(item)
	        print query
	        num_results_returned = 0
	        if query != "":
	            try:
	            	query_assembled = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/v1/Web?Query=%27"+query+"%27&$format=json&$top=" + str(num_api_return)
	                search = requests.get(query_assembled, auth=("", "*** add your Bing API authorization code here ***")).json
	                
	                # store returned URLs(e.g., 5 urls per sentence) into a list
	                results = search['d']['results']
	                # combine all returned results into a single string and append it
	                group_str_0 = ""
	                group_str_1 = ""

					# keep track of the number of results for each API query
	                num_results_returned = len(results)
					# for M_39 configuration (50 results merged) -----------------//
	                for result in results:
	                    clean_result = result['Description'].replace("...", "").strip().replace("\"", "")
	                    # clean_result = " ".join(self.parseInput(result['Description']))
	                    # print clean_result
	                    group_str_0 = group_str_0 + " " + clean_result
	                micro_corpus_0.append(group_str_0)

	                # for M_43 configuration (only 10 results merged) ------------//
	                for result in results[0:10]:
	                    clean_result = result['Description'].replace("...", "").strip().replace("\"", "")
	                    # clean_result = " ".join(self.parseInput(result['Description']))
	                    # print clean_result
	                    group_str_1 = group_str_1 + " " + clean_result
	                micro_corpus_1.append(group_str_1)

	                # add both micro_corpus_0 and micro_corpus_1 to micro_corpus
	                micro_corpus.append(micro_corpus_0)
	                micro_corpus.append(micro_corpus_1)

	            except Exception, err:
	                #sys.stderr.write('ERROR: %s\n' % str(err))
	                print "Something wrong............"

	                pass
                # micro_corpus.append(micro_corpus_0)
                # micro_corpus.append(micro_corpus_1)

	        else:
	        	num_results_returned = 0

	        num_results_returned_li.append(num_results_returned)

	    # print "micro_corpus_1---------------------------"
	    # print micro_corpus_1

	    micro_corpus.append(micro_corpus_0)
	    micro_corpus.append(micro_corpus_1)
	    return micro_corpus

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

	# # tf*idf weighting applied to self.A matrix                 
	# def tfidf(self):
	# 	WordsPerDoc = sum(self.A, axis=0)        
	# 	DocsPerWord = sum(asarray(self.A > 0, 'f'), axis=1)
	# 	rows, cols = self.A.shape
	# 	for i in range(rows):
	# 		for j in range(cols):
	#             # multiply IDF after normailizing TF
	# 			self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i]) 

	# # tf*df weighting applied to self.A matrix                 
	# def tfdf(self):
	# 	WordsPerDoc = sum(self.A, axis=0)        
	# 	DocsPerWord = sum(asarray(self.A > 0, 'f'), axis=1)
	# 	rows, cols = self.A.shape
	# 	for i in range(rows):
	# 		for j in range(cols):
	# 			# self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i]) 
	# 			# self.A[i,j] = self.A[i,j]*float(DocsPerWord[i])/float(cols) 
	# 			self.A[i,j] = self.A[i,j]*float(DocsPerWord[i])/float(cols) 

	# def df(self):
	# 	WordsPerDoc = sum(self.A, axis=0)        
	# 	DocsPerWord = sum(asarray(self.A > 0, 'f'), axis=1)
	# 	rows, cols = self.A.shape
	# 	for i in range(rows):
	# 		for j in range(cols):
	# 			# self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i]) 
	# 			# self.A[i,j] = self.A[i,j]*float(DocsPerWord[i])/float(cols) 
	# 			self.A[i,j] = float(DocsPerWord[i])/float(cols) 

	# # itf*df weighting applied to self.A matrix                 
	# def itfdf(self):
	# 	WordsPerDoc = sum(self.A, axis=0)        
	# 	DocsPerWord = sum(asarray(self.A > 0, 'f'), axis=1)
	# 	rows, cols = self.A.shape
	# 	for i in range(rows):
	# 		for j in range(cols):
	#             # multiply IDF after normailizing TF
	# 			# self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i]) 
	# 			# itf * df
	# 			if self.A[i,j] != 0:
	# 				self.A[i,j] = log(WordsPerDoc[j]/self.A[i,j]) * float(DocsPerWord[i])/float(cols) 
	# 			else:
	# 				self.A[i,j] = 0.0

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
	num_topics = 20
	window_overlap = 1
	# mysql credentials ------------------------------ (MySQL)
	# ------------------------------------- (various_30)
	# dbuser = "db username"
	# dbpasswd = "db password"
	# hostname = "db hostname"
	# dbname = "various_30"
	# dbtable = "machine_lem_tfdf"
	# # dbtable = "machine_lem_itfdf"

	# ------------------------------------- (ctr_30)
	# dbuser = "db username"
	# dbpasswd = "db password"
	# hostname = "db hostname"
	# dbname = "ctr_30"
	# dbtable = "machine_lem_tfdf"
	# dbtable = "machine_lem_itfdf"

	# ------------------------------------- (nyt_3000)
	dbuser = "db username"
	dbpasswd = "db password"
	hostname = "db hostname"
	dbname = "nyt"
	dbtable = "nyt_3000"
	# develop id list
	fi = open("NYT_1000_IDs_EXP3.txt", "r")
	li = fi.read().split()
	fi.close()

    # connect to mysql
	db = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
	cur = db.cursor()
	# input text ------------------------------------- (Control inputs)
	iter_id = 1
	for u_size in [20, 15, 10, 5, 1][3:4]:
		for a_return in [50, 10, 1][0:1]:
			# for t_idf in [1, 0][1:2]:
			for tfdf in [1, 0][1:2]:
			# for itfdf in [1, 0][0:1]:
				for o_nouns in [1, 0][0:1]:  # always use nouns only
					# -------------------------------- (Process starts)
					for filenum in li:  # id: 1-1000 NYT articles 
						text = ""
						doc_id = filenum
						print "Document ID: %s is being processed.........\n" % doc_id
						filename = str(filenum) + ".txt"
						# text = open("..\\text_various_30\\"+filename, "r").read()
						text = open("..\\NYT_3000\\"+filename, "r").read()   
						# text = open("..\\ctr_30\\"+filename, "r").read()   # to run in Windows
						# text = open("../NYT_3000/"+filename, "r").read()   
						# text = open("../text_various_30/"+filename, "r").read()	   # to run in Linux/MacOSX
						# text = open("test.txt", "r").read()
						#-----------------------------
						xpt_0 = Xpantrac("stopwords.txt", "custom_stops.txt")
						xpt_1 = Xpantrac("stopwords.txt", "custom_stops.txt")
						parsed_input_li = xpt_0.parseInput(text)
						query_list = xpt_0.makeQueryUnits(parsed_input_li, u_size, window_overlap)
						micro_corpus = xpt_0.makeMicroCorpus(query_list, a_return)
						# get only nouns, plural nouns, and proper nouns
						if o_nouns:
							xpt_0.parse2Dict(micro_corpus[0], ["NN", "NNS", "NNP"])
							xpt_1.parse2Dict(micro_corpus[1], ["NN", "NNS", "NNP"])
						else:
							xpt_0.parse2Dict(micro_corpus[0])
							xpt_1.parse2Dict(micro_corpus[1])
						xpt_0.build()
						xpt_1.build()

						# if tfdf:
						# # if itfdf:
						# 	print "\n------------- doc: %s --------------\n" % filename
						# 	print text
						# 	print "\n----------------- END --------------\n"

						# 	xpt.tfdf()
						# 	# xpt.itfdf()

						print "------- m39 ---------"
						xpt_0.build_sum_dict()
						m_39_topics = xpt_0.extTopics(num_topics)
						print "------- m43 ---------"
						xpt_1.build_sum_dict()
						m_43_topics = xpt_1.extTopics(num_topics)

						# get intersection and union of m_39, m_43
						m_39_li = m_39_topics.split(',')
						m_43_li = m_43_topics.split(',')

						topic_intersection = computeIntersection(m_39_li, m_43_li)
						topic_inter_str = ",".join(topic_intersection)
						topic_union = computeUnion(m_39_li, m_43_li)
						topic_uni_str = ",".join(topic_union)
						
						print "-----------------------"
						print doc_id
						print m_39_topics
						print m_43_topics
						print topic_inter_str
						print topic_uni_str
						num_return_count_li = []
						for i in micro_corpus[-1]:
							num_return_count_li.append(str(i))
						num_api_results_str = ",".join(num_return_count_li)
						print num_api_results_str
						print "-----------------------"

					    # write to mysqldb       
						query = "update "+ dbtable +" set m_39_bing_100doc_20tag='" + m_39_topics + "', m_43_bing_100doc_20tag='" + m_43_topics + "', 39_AND_43_bing_100doc_20tag='" + topic_inter_str + "', 39_OR_43_bing_100doc_20tag='" + topic_uni_str + "', num_api_results_bing='" + num_api_results_str + "' where doc_id=" + str(doc_id)
						cur.execute(query)
					# --------------------------- (Update iteration)
					iter_id += 1

if __name__ == "__main__":
	main()