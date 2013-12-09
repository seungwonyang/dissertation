#!/usr/local/bin/env python2.7
# Name: Seungwon Yang  July. 6, 2013
# Filename: compute_IIC_human_machine.py
# Description: this script first develops a centroid vector using all
#              human assigned topics. Then, it computes cosine similarity
#              between each human topic group and the centroid, which can be
#			   averaged.  Cos similarity between machine topic group and the
#              centroid also is computed.
#              For each document, human_ave similarity and machine similarity
#              are acquired (for independent two-sample t-test to see if
#              there are any differences between human vs machine topics)
# Note: Bing Azure search API is used.
# Usage: #>python compute_IIC_human_machine.py

import sys
import re
import codecs
import json
import logging
import MySQLdb
import requests
import string
import urllib2
from math import log, sqrt
from nltk import pos_tag, word_tokenize, stem
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

	def makeMicroCorpus(self, host, user, passwd, db, dbtable, dbtable2, machine_id, human_id_li):
		db = MySQLdb.connect(host, user, passwd, db)
		# create a cursor 
		cur = db.cursor()
		# dbtable = "human_lem"  # Hurricane Isaac tweet collection
		micro_corpus = []
		for i in range(1,31):
			corpus_per_topics = []
			# for j in [1,3,4]:
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

def main():
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	host = "spare05.dlib.vt.edu"

	# ------------------------------ CTR_30 DB access account --------------- #
	# user = "username"
	# passwd = "password"
	# db = "ctr_30"
	# dbtable = "human_lem"
	# dbtable2 = "machine_lem"
	# human_id_li = [1,3,4]  # indexers 1,2, and 3

	# ------------------------------ VARIOUS_30 DB access account----------- #
	user = "username"
	passwd = "password"
	db = "various_30"
	dbtable = "human_lem"	
	dbtable2 = "machine_lem"
	human_id_li = [11,12,13,14,15] # indexers 11, 12,13,14, and 15

	for machine_id in range(39,40): # compute IIC for M39 configuration
		print "\nmachine_id: %d ------------------" % machine_id
		cs = CosSim("stopwords.txt", "custom_stops.txt")
		micro_corpus = cs.makeMicroCorpus(host, user, passwd, db, dbtable, dbtable2, machine_id, human_id_li)
		# print micro_corpus	
		keepNum = 1
		for i in range(0, len(human_id_li)):
			for j in range(i+1, len(human_id_li)+1): # include M topics
				rollingSum = 0
				rollingLi = []
				rollingDiff = []
				print "(%d, %d)" % (i,j)
				for doc in micro_corpus:
					setA = cs.removeSymLemmatize(doc[i])
					setB = cs.removeSymLemmatize(doc[j])
					# print "set A and B -----------"
					# print set(setA)
					# print set(setB)
					rollingScore = cs.computeRolling(setA, setB)
					rollingLi.append(rollingScore)
					# print rollingScore
					# rollingSum = rollingSum + rollingScore
				rollingSum = sum(rollingLi)
				rollingAve = float(rollingSum)/float(30)
				rollingDiffSquare = [(roll-rollingAve)*(roll-rollingAve) for roll in rollingLi]
				rollingStddev = sqrt(sum(rollingDiffSquare)/float(len(rollingLi)))
				print "Rolling average: %0.3f	stddev:%0.3f -------\n" % (rollingAve, rollingStddev)

			keepNum += 1

if __name__ == "__main__":
	main()