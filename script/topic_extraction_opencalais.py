# Name: Seungwon Yang  July 31, 2013
# Filename: topic_extraction_opencalais.py
# Description: this script extracts OpenCalais topic tags provided a text dataset.
# Usage: #>python topic_extraction_opencalais.py
# Note: the filenames and their directory are hard-coded in the script. Change them for your purpose. Add your database access information as well.  Also, you need an API key from opencalais.com.

import sys
import re
import codecs
import json
import logging
import MySQLdb
import requests
import string
import time
import urllib2
from calais import Calais

class OpCal:
	# make a stopword set
	def __init__(self, stop_path, custom_stop_path):
		stop_li = open(stop_path, "r").read().split()
		custom_stop_li = open(custom_stop_path, "r").read().split()
		self.stoplist = list(set(stop_li + custom_stop_li))

	# this function removes symbol characters, and ignores foreign languages
	def removeSymbols(self, any_text):	
		# first, decode already encoded text
		utf8_str = any_text.decode('utf-8')
		# then, encode with ascii only
		only_ascii = utf8_str.encode('ascii', 'ignore')
		# remove symbol characters
		return re.sub(r'[^\w]', ' ', only_ascii)		

	def splitCalaisTopics(self, calais_topic_li):
		split_space_li = [elem for sublist in [item.lower().split()  for item in calais_topic_li] for elem in sublist]
		split_underscore_li = [elem for sublist in [item.split('_') for item in split_space_li] for elem in sublist]
		return [item for item in list(set(split_underscore_li)) if item not in self.stoplist]	
	
def main():
	
	oc = OpCal("stopwords.txt", "custom_stops.txt")
	hostname = "add.your.hostname_to_db.here"
	# --------------------- DB account info ------------------------------ #
	dbuser = "db username"
	dbpasswd = "db password"
	dbname = "db name"
	dataset = "document folder name"
	dbtable = "db table name to insert opencalais topics"

	# --------------------- Connect to DB -------------------------------- #
	db = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
	cur = db.cursor()	
	# -------------------------------------------------------------------- #	

	# --------------------- Prep for OpenCalais processing --------------- #
	API_KEY = "add your OpenCalais API key here"
	calais = Calais(API_KEY, submitter="topic extractor")

	# ------------------- OpenCalais topic extraction BEGIN -------------- #
	# Note: Not more than 4 API calls per second
	#       50,000 rate limit per day
	# -------------------------------------------------------------------- #
	
	for i in range(1,31):  # change appropriately based on the pattern of your filename
		calais_topics = []
		path = "../" + dataset + "/" + str(i) + ".txt"
		indoc = open(path, "r").read()
		print "Processing____%s: Doc: %d ____" % (dataset, i)
		clean_txt = oc.removeSymbols(indoc)
		result = calais.analyze(clean_txt)
		# extract topics from OpenCalais
		topic_li = result.print_topics_clean()
		# extract named entities from OpenCalais
		entity_li = result.print_entities_clean()
		if topic_li != None:
			calais_topics = calais_topics + topic_li
		if entity_li != None:
			calais_topics = calais_topics + entity_li
		time.sleep(0.5)  # to abide by the terms of service
		topics_str = ",".join(oc.splitCalaisTopics(calais_topics))
		print "---- split calais topics ----\n"
		print topics_str
	    # write to mysqldb -- change it to fit your db schema       
		query = "insert into "+ dbtable + " (doc_id, topic) values('" + str(i) + "','" + topics_str + "')"   
		cur.execute(query)		
	# ------------------- OpenCalais topic extraction END -------------- #
	
if __name__ == "__main__":
	main()