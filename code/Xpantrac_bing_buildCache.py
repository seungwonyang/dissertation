# Name: Seungwon Yang  July 28, 2013
# Filename: Xpantrac_bing_buildCache.py
# Description: this script expands an input text using the Bing Azure API, and then
#              stores the expanded informaiton to a database table for later topic extraction.
# Note: Bing search API is used.
# Usage: #>python Xpantrac_bing_buildCache.py
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
		# remove symbol characters
		return re.sub(r'[^\w]', ' ', only_ascii)		
	
	def parseInput(self, text):
		clean_text = self.removeSymbols(text).lower()
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
	    # for item in query_list[:5]: # [[query unit 1], [query unit 2], [query unit 3],...]
	    for item in query_list: # [[query unit 1], [query unit 2], [query unit 3],...]
	        query = " ".join(item)
	        print query
	        num_results_returned = 0
	        if query != "":
	            try:
	            	# query_assembled = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/v1/Web?Query=%27"+query+"%27&$format=json&$top=" + str(num_api_return)
	                
	                query_assembled = "https://api.datamarket.azure.com/Bing/SearchWeb/v1/Web?$format=json&Query=%27"+query+"%27&$top=" + str(num_api_return)
	                # ---- using topic_ui@hotmail.com account ------------------########
	                search = requests.get(query_assembled, auth=("", "*** add here an authrization code for Bing API")).json
	                
	                results = search['d']['results']
	                # combine all returned results into a single string and append it
	                group_str_0 = ""
	                group_str_1 = ""

					# keep track of the number of results for each API query
	                num_results_returned = len(results)
					# for M_39 configuration (50 results merged) -----------------//
	                clean_result_li = []
	                for result in results:
	                    clean_result = result['Description'].replace("...", "").strip().replace("\"", "")
	                    clean_result_li.append(clean_result)

	                    # clean_result = " ".join(self.parseInput(result['Description']))
	                    # print clean_result
	                    # group_str_0 = group_str_0 + " " + clean_result
	                micro_corpus_0.append(clean_result_li)
	                # add both micro_corpus_0 and micro_corpus_1 to micro_corpus
	                micro_corpus.append(micro_corpus_0)
	                micro_corpus.append(micro_corpus_1)

	            except Exception, err:
	                sys.stderr.write('ERROR: %s\n' % str(err))
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
	    micro_corpus.append(num_results_returned_li)

	    # micro_corpus = [corpus_50, corpus_10, api_return_count_li]
	    return micro_corpus

def main():
	# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	# parameters ------------------------------------ (Overall)
	# unit_size = 10	
	# num_api_return = 10
	num_topics = 20
	window_overlap = 1

	# mysql credentials ------------------------------ (MySQL)

	# ------------------------------------- (various_30)
	dbuser = "db username"
	dbpasswd = "db password"
	hostname = "db hostname"
	dbname = "various_30"
	dbtable = "api_results_50_bing"
	# dbtable = "machine_lem_itfdf"

	# ------------------------------------- (ctr_30)
	# dbuser = "db username"
	# dbpasswd = "db password"
	# hostname = "db hostname"
	# dbname = "ctr_30"
	# dbtable = "api_results_50_bing"
	# # dbtable = "machine_lem_itfdf"

	# # ------------------------------------- (nyt_3000)
	# dbuser = "db username"
	# dbpasswd = "db password"
	# hostname = "db hostname"
	# dbname = "nyt"
	# # dbtable = "nyt_3000"
	# dbtable = "api_results_50_bing"

	# api = "yahooweb"     
	# api = "yahoonews"     
	api = "bing" 
	doc_content = "0"

	# # develop id list
	# fi = open("NYT_3000_IDs.txt", "r")
	# li = fi.read().split()
	# fi.close()
	
    # connect to mysql
	db = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
	cur = db.cursor()
	# ind = 3
	# for nyt_id in li[2:]:
	# 	print ind
	# 	query = "insert into "+ dbtable + " (doc_id, m_39_topics, m_43_topics, 39_AND_43, 39_OR_43) values('" + nyt_id + "','" + "" + "','" + "" + "','" + "" + "','" + "" + "')"   
	# 	cur.execute(query)
	# 	ind += 1

	# input text ------------------------------------- (Control inputs)
	iter_id = 1
	for u_size in [20, 15, 10, 5, 1][3:4]:
		for a_return in [50, 10, 1][0:1]:
			# for t_idf in [1, 0][1:2]:
			for tfdf in [1, 0][1:2]:
			# for itfdf in [1, 0][0:1]:
				for o_nouns in [1, 0][0:1]:  # always use nouns only
					# -------------------------------- (Process starts)
					for filenum in range(1,31):  # id: 1-30 ctr_30 articles 
						text = ""
						doc_id = filenum
						print "\n------- Document ID: %s is being processed ---------\n" % doc_id
						filename = str(filenum) + ".txt"
						# text = open("..\\text_various_30\\"+filename, "r").read()
						# text = open("..\\NYT_3000\\"+filename, "r").read()   
						# text = open("..\\ctr_30\\"+filename, "r").read()   # to run in Windows
						# text = open("../ctr_30/"+filename, "r").read()
						# text = open("../NYT_3000/"+filename, "r").read()   
						   
						text = open("../various_30/"+filename, "r").read()	   # to run in Linux/MacOSX
						# text = open("test.txt", "r").read()

						#-----------------------------
						# xpt_0: num_api_return = 50
						# xpt_1: num_api-return = 10 
						xpt_0 = Xpantrac("stopwords.txt", "custom_stops.txt")
						xpt_1 = Xpantrac("stopwords.txt", "custom_stops.txt")

						parsed_input_li = xpt_0.parseInput(text)
						query_list = xpt_0.makeQueryUnits(parsed_input_li, u_size, window_overlap)
						micro_corpus = xpt_0.makeMicroCorpus(query_list, a_return)
						# print len(micro_corpus[0])
						# print micro_corpus[1]

						query_id = 1
						for corp in micro_corpus[0]:
						# for each_desc in micro_corpus[0]:
							# reconstruct a clean result li
							no_sym_results_li = []
							for each_desc in corp:
								no_sym_results_li.append(re.sub(r'[^\w]', ' ', each_desc))

							results_str = str(no_sym_results_li) # 50 results list as string
							query_str = " ".join(query_list[query_id - 1])

							# new_doc_id = "ctr_%d" % doc_id
							new_doc_id = "various_%d" % doc_id
							print new_doc_id
							print query_id
							print query_str
							# print results_str
							print api
							print doc_content
							# write to mysql
							query = "INSERT INTO "+ dbtable + " (doc_id, query_id, query, api, results, doc_content) values('" + str(new_doc_id) + "'," + str(query_id) + ",'" + query_str + "','" + api + "',\"" + results_str + "\",'" + str(doc_content) + "')"
							cur.execute(query)							
							query_id += 1
					# --------------------------- (Update iteration)
					iter_id += 1

if __name__ == "__main__":
	main()