# Name: Seungwon Yang  Aug. 4, 2013
# Filename: topic_extraction_TFIDF_NYT_1000.py
# Description: this is to compute TF*IDF-only topic tags from the
#              NYT 1000 doc dataset.
# Usage: #>python topic_extraction_TFIDF_NYT_1000.py

import sys
import re
import codecs
import json
import logging
import matplotlib.pyplot as plot
import MySQLdb
import requests
import string
import urllib2
from math import log, sqrt
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import *
from operator import itemgetter # to sort dictionaries by their keys

class CosSim:
	# make a stopword set
	def __init__(self, stop_path, custom_stop_path):
	# def __init__(self):
		stop_li = open(stop_path, "r").read().split()
		custom_stop_li = open(custom_stop_path, "r").read().split()
		self.stoplist = set(stop_li + custom_stop_li)
		# self.stoplist = []
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

	def parse2Dict(self, micro_corpus, nounsOnly):
		# words = " ".join(micro_corpus).split();
		lemm = WordNetLemmatizer()
		for doc in micro_corpus:
			doc_list = []
			# doc = self.removeSymbols(doc)

			# check nounsOnly value
			if nounsOnly != []:
				pos_tag_li = pos_tag(word_tokenize(doc))
				for key,val in pos_tag_li:
					if val in nounsOnly:
						# doc_no_symbol = re.sub(r'[^\w]', ' ', key)
						# doc_list.append(doc_no_symbol.lower())
						doc_list.append(key.lower())
			else:
				doc_no_symbol = re.sub(r'[^\w]', ' ', doc)
				doc_list = doc_no_symbol.lower().split() 
				# doc_list = key.lower().split()
			for w in doc_list:
				wlem = lemm.lemmatize(w)
				if (wlem in self.stoplist) or (len(wlem) < 3):
					continue
				elif wlem in self.wdict:
					self.wdict[wlem].append(self.dcount)
				else:
					self.wdict[wlem] = [self.dcount]
			self.dcount += 1   
		# print self.wdict

	def parse2HumanDict(self, each_corpus, nounsOnly=[]):
		# words = " ".join(micro_corpus).split();
		for doc in each_corpus[0:-1]: # use only human indexer topics for dic
			doc_list = []

			# check nounsOnly value
			if nounsOnly != []:
				pos_tag_li = pos_tag(word_tokenize(doc))
				for key,val in pos_tag_li:
					if val in nounsOnly:
						doc_list.append(key.lower())
			else:
				doc_no_symbol = re.sub(r'[^\w]', ' ', doc)
				doc_list = doc_no_symbol.lower().split() 

			for w in doc_list:
				if (len(w) < 2):
					continue
				elif w in self.wdict:
					self.wdict[w].append(self.dcount)
				else:
					self.wdict[w] = [self.dcount]
			self.dcount += 1   

		# at this point, self.wdict is constructed using only human indexer topics

		# it's time to add machine topic presence in self.wdict terms
		machine_doc_id = self.dcount  # since human indexer starts from '0'
		machine_doc = each_corpus[self.dcount]
		machine_doc_list = []
		# check nounsOnly value
		if nounsOnly != []:
			pos_tag_li = pos_tag(word_tokenize(machine_doc))
			for key,val in pos_tag_li:
				if val in nounsOnly:
					machine_doc_list.append(key.lower())
		else:
			doc_no_symbol = re.sub(r'[^\w]', ' ', machine_doc)
			machine_doc_list = doc_no_symbol.lower().split() 

		for w in machine_doc_list:
			if (len(w) < 2):
				continue
			elif w in self.wdict:
				self.wdict[w].append(machine_doc_id)
			else:
				continue  #if a machine topic not exist in self.wdict, skip adding it to wdict
				# self.wdict[w] = [self.dcount]
		self.dcount += 1   # increment to add machine topics with human topics

	# this function develops a micro-corpus using the entire docs in
	# either ctr_30 or various_30.
	# Symbol chars, and stop words are not included in the corpus docs.
	
	def makeMicroCorpus(self, host, user, passwd, db, dbtable, dbtable2, machine_id, human_id_li):
		db = MySQLdb.connect(host, user, passwd, db)
		# create a cursor 
		cur = db.cursor()
		# dbtable = "human_lem"  # Hurricane Isaac tweet collection
		micro_corpus = []
		for i in range(1,31):
			# print "------------topics: %d ------------\n" % i
			# query = "update "+ dbtable +" set topics" + str(i) + "= replace(topics" + str(i) +", ',', ' ')"
			# query = "update "+ dbtable +" set topics" + str(i) + "=LOWER(topics" +str(i) + ")"
			# cur.execute(query)
			corpus_per_topics = []
			for j in human_id_li:
			  query = "select topics" + str(i) + " from "+dbtable+" where participant_id=" + str(j)
			  cur.execute(query)
			  single_doc_in_corpus = " ".join(cur.fetchone()[0].split(","))
			  corpus_per_topics.append(single_doc_in_corpus)
			# get topics data from machine
			query2 = "select topics" + str(i) + " from "+dbtable2+" where id=" + str(machine_id)
			cur.execute(query2)
			machine_doc_in_corpus = " ".join(cur.fetchone()[0].split(","))
			corpus_per_topics.append(machine_doc_in_corpus)
			micro_corpus.append(corpus_per_topics)
		return micro_corpus

	# add TF (term frequency) in each cell of the term-document matrix
	def build(self):
		# self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
		self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 0]
		self.keys.sort()
		num_keys = len(self.keys)
		# min_cell_val = 1 / float(num_keys)
		# create a zero matrix of terms (i.e., keys) and document numbers (i.e., columns)
		self.A = zeros([len(self.keys), self.dcount])  # include machine doc column 
		# update cell values with wdict values
		for i, k in enumerate(self.keys):
			for d in self.wdict[k]:               
				self.A[i,d] = self.A[i,d] + 1
	
	# tf*idf weighting applied to self.A matrix                 
	def tfidf(self):
		WordsPerDoc = sum(self.A, axis=0)        
		DocsPerWord = sum(asarray(self.A > 0, 'f'), axis=1)
		rows, cols = self.A.shape
		for i in range(rows):
			for j in range(cols):
	            # multiply IDF after normailizing TF
				self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i]) 

	# this function adds all cell values for each key (i.e., row)
	def build_sum_dict(self, doc_num):
	    # initialize sum_dict
	    self.sum_dict = {}
	    rows, cols = self.A.shape 
	    # print "self.A ----------"
	    # print self.A
	    for i, k in enumerate(self.keys):
	        self.sum_dict[k] = self.A[i, doc_num]     
	    # self.sum_dict = self.A[:, doc_num:doc_num+1]
	    # print self.sum_dict

	def printTopics(self, num_topics):		
	    # sort self.sum_dict in descending order
		sorted_list = sorted(self.sum_dict.iteritems(), key=itemgetter(1), reverse=True)
	    # select top num_topics and print them out
		topics_li = []
		topics_str = ""
		# for item, score in sorted_list[:num_topics]:
		# 	topics_li.append(item)
		# topics_str = ",".join(topics_li)
		# print topics_str
		for item in sorted_list[:num_topics]:
			print item

	def printTopics2Db(self, num_topics, dbuser, dbpasswd, hostname, dbname, dbtable, input_id):
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
	    # write to mysqldb       
		# query = "insert into "+ dbtable + " (id,unit_size,num_api_return,tfidf,only_nouns,topics" + str(input_doc_id) + ") values('" + str(iter_id) + "','" + str(unit_size) + "','" + str(num_api_return) + "','" + str(tfidf) + "','" + str(only_nouns) + "','" + topics_str + "')"   

		query = "update "+ dbtable +" set TFIDF='" + topics_str + "' where id=" + str(input_id)

		cur.execute(query)		
	
	def calc_cosine_sim(self, u, v):
		return dot(u, v) / (sqrt(dot(u,u)) * sqrt(dot(v,v)))

	# this function calculates similarity score between the original input doc
	# and a set of website descriptions expanded for each query unit
	# tf*idf_C function is applied before similarity calculations
	def create_sim_list(self, topics_index):
		self.B = self.A
		rows, cols = self.A.shape
		centroid_vec = zeros(rows)
		num_keys = len(self.keys)

		# centroid_vec construction using human and machine topics
		centroid_vec = sum(asarray(self.A > 0, 'f'), axis=1)
		for i in range(0, len(centroid_vec)):
			centroid_vec_val = centroid_vec[i]
			centroid_vec[i] = float(centroid_vec_val)/float(cols)
			# print centroid_vec[i]
		centroid_vec.shape = (rows, 1)
		
		self.B = append(self.B, centroid_vec, 1)
		rows, cols = self.B.shape
		self.sim_list = []
		cos_sim_val_sum = 0
		for doc_num in range(cols-1):
			cos_sim_val = self.calc_cosine_sim(self.B[:,doc_num],self.B[:,cols-1])
			self.sim_list.append(cos_sim_val)
			cos_sim_val_sum += cos_sim_val
		return self.sim_list

	def create_sim_list_2(self, topics_index):
		self.B = self.A
		rows, cols = self.A.shape
		centroid_vec = zeros(rows)
		num_keys = len(self.keys)

		# centroid_vec construction using only human topics
		centroid_vec = sum(asarray(self.A[:,:cols-1] > 0, 'f'), axis=1)
		for i in range(0, len(centroid_vec)):
			centroid_vec_val = centroid_vec[i]
			centroid_vec[i] = float(centroid_vec_val)/float(cols-1)
			# print centroid_vec[i]
		centroid_vec.shape = (rows, 1)
		
		# compute sim values between each human topic group and the centroid.
		self.B = append(self.B, centroid_vec, 1)
		rows, cols = self.B.shape
		self.sim_list = []
		cos_sim_val_sum = 0
		for doc_num in range(cols-1):
			cos_sim_val = self.calc_cosine_sim(self.B[:,doc_num],self.B[:,cols-1])
			self.sim_list.append(cos_sim_val)
			cos_sim_val_sum += cos_sim_val

		return self.sim_list

	def plot_indiv_sim(self, sim_list, machine_id, topics_id):
		mean_val = mean(sim_list)
		n = len(sim_list)
		X = arange(n)
		Y = sim_list
		
		X2 = arange(n+1)
		Y2 = [mean_val for x in X2]
		width = 0.4

		fig = plot.figure()
		ax = fig.add_subplot(111)
		rects = ax.bar(X, Y, facecolor='#9999ff', edgecolor='white')
		rects2 = ax.plot(X2, Y2, color='r', linewidth=1.0)
		
		for x,y in zip(X, Y):
			ax.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
		ax.set_ylim(0, +1.1)
		ax.set_xticklabels(('h_1','h_2','h_3','m'))
		ax.set_xlabel("Indexer")
		ax.set_ylabel("Cosine similarity [0, 1]")
		ax.set_title("Inter-Indexer Consistency (mean: " + str(mean_val) +")")
		ax.set_xticks(X+width)
		ax.set_xticklabels(('h1','h2','h3','m'))
		filename = "human_machine_sim_graphs/humans_machine_id_" + str(machine_id) + "_topics_id_"+str(topics_id)+".png" 
		plot.savefig(filename)
		plot.clf()
		# bar.clf()
		# plot.show()

	def multi_plot_sim_lists(self, sim_list_all):
		mean_val = []
		std_val = []
		for item in sim_list_all:
			mean_val.append(mean(item))
			std_val.append(std(item))
		n = len(mean_val)
		X = np.arange(n)
		Y = mean_val
		mean_of_means = mean(mean_val)
		X2 = np.arange(n+1)
		Y2 = [mean_of_means for x in X2]
		plot(X2, Y2, color='r', linewidth=1.0)
		rects = bar(X, Y, facecolor='#9999ff', edgecolor='white', yerr=std_val)
		for x,y in zip(X, Y):
			text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')
		ylim(0, +1.15)
		
		xlabel("Topic number")
		ylabel("ICD [0, 1]")
		# title("Human Inter-Indexer Consistency using Cosine Similarity (mean: " + str(mean_of_means) +")")
		title("Human Inter-Indexer Consistency Density (ICD) based on Cosine Similarity (mean: " + str(mean_of_means) +")")
		savefig("human_ave_sim_graphs_30.png")
		# show()
		
	def print_B(self):
		print "------- Term-Doc matrix B --------"
		for item in self.B:
			print item

	def print_A(self):
		print "------- Term-Doc matrix A --------"
		for item in self.A:
			print item

def main():
	num_topics = 20
	# nounsOnly=["NN", "NNS", "NNP"]
	obj = CosSim("stopwords.txt", "custom_stops.txt")
	# develop doc_id list
	fi = open("NYT_1000_IDs_EXP3.txt", "r") # get the IDs of NYT articles
	li = fi.read().split()
	fi.close()

	dbuser = "username"
	dbpasswd = "password"
	hostname = "host.name.of.database"
	dbname = "nyt"
	dbtable = "nyt_1000"
	dataset = "NYT_3000"
	# make a corpus
	micro_corpus = []
	i = 1
	# for i in range(1,31):
	for filenum in li[0:1000]:
		# NYT data set (1000 docs)
		filename = str(filenum) + ".txt"
		# text = open("..\\"+dataset+"\\"+filename, "r").read()   
		indoc = open("../"+dataset+"/"+filename, "r").read()
		print "Processing____%s: Doc: %d ____" % (dataset, i)
		micro_corpus.append(obj.removeSymbols(indoc))
		i += 1

	# ---- Develop a dictionary using micro_corpus --------------------- #
	# obj.parse2Dict(micro_corpus, nounsOnly)
	obj.parse2Dict(micro_corpus, [])

	# ---- Construct a term-document matrix ---------------------------- #
	obj.build()

	# ---- Apply TF*IDF weighting -------------------------------------- #
	obj.tfidf()

	# ---- Select top n words as topics -------------------------------- #
	for i in range(1,1001):
	# loop through each column, sort by the cell value in desc. order 
		obj.build_sum_dict(i-1)
		print "\n--------- Topics for Doc:[%d] ---------" % (i)
		# print topics per each document
		# obj.printTopics(num_topics)
		obj.printTopics2Db(num_topics, dbuser, dbpasswd, hostname, dbname, dbtable, i)

if __name__ == "__main__":
	main()