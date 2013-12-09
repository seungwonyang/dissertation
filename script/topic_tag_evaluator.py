# Filename: topic_tag_evaluator.py
# Description: This script extracts topic tags from MySQL database, and
#              evaluates tags using gold standard tags, which is also stored in a DB.
#              
# shell>python topic_tag_evaluator.py
# Name: Seungwon Yang  <seungwon@vt.edu>
# Date: July 29, 2013

import base64
import sys 
import string
import re
import codecs
import urllib
import urllib2
import datetime
import MySQLdb
import os
import sets
from nltk.stem.wordnet import WordNetLemmatizer

def removeSymbols(word):
  return re.sub(r'[^\w]', ' ', word)

def computeIntersection(a_li, b_li):
    return list(set(a_li) & set(b_li))
  
def computeMetric(cur, dbtable, metric_type):
  # print "---------------------------------------------------"
  # print "   Precision     Recall       F-1 "
  # print "---------------------------------------------------"

  metric_li_li = []
  # for item in range(1, 1501):
  for item in range(1, 1001):  # id range [1, 1000] 
  # for item in range(1, 10):

    # make gold standard
    query0 = "select nyt_manual_topics_lem from " + dbtable + " where id='" + str(item) + "'"
    cur.execute(query0)
    gold_standard_li = cur.fetchone()[0].split(",")
    len_gold_standard = len(gold_standard_li)

    tag_li = ['M39_bing', 'M39_yahooweb', 'M39_yahoonews', 'M43_bing', 'M43_yahooweb', 'M43_yahoonews', '39_AND_43_bing', '39_AND_43_yahooweb', '39_AND_43_yahoonews', '39_OR_43_bing', '39_OR_43_yahooweb', '39_OR_43_yahoonews', 'opencalais_lem_split', 'TFIDF']
    
    metric_li = []
    for tag in tag_li:
      query = "select " + tag + " from " + dbtable + " where id='" + str(item) + "'"
      cur.execute(query)
      topic_groups = cur.fetchone()
      each_topic_li = topic_groups[0].split(",")
      intersection = computeIntersection(each_topic_li, gold_standard_li)
      len_each_topic_li = len(each_topic_li)
      len_intersection = len(intersection)
      
      metric = 0.0
      if metric_type == "p":
        # precision
        metric = float(len_intersection) / len_each_topic_li
      elif metric_type == "r":
        # recall
        metric = float(len_intersection) / len_gold_standard
      elif metric_type == "f1":
        # precision
        P = float(len_intersection) / len_each_topic_li
        # recall
        R = float(len_intersection) / len_gold_standard

        metric = 0.0
        denominator = (float(P) + float(R))
        if denominator != 0.0:
          metric = float(2)*P*R / denominator


      else:
        "Please double check the metric type in function arguments"

      metric_li.append(metric)
    metric_li_li.append(metric_li)
  print "index\tM39_bing\tM39_yahooweb\tM39_yahoonews\tM43_bing\tM43_yahooweb\tM43_yahoonews\tM39_AND_M43_bing\tM39_AND_M43_yahooweb\tM39_AND_M43_yahoonews\tM39_OR_M43_bing\tM39_OR_M43_yahooweb\tM39_OR_M43_yahoonews\tOpencalais\tTFIDF"
  
  ii = 1
  for met_li in metric_li_li:
    print "%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (ii, met_li[0],met_li[1],met_li[2],met_li[3],met_li[4],met_li[5],met_li[6],met_li[7],met_li[8],met_li[9],met_li[10],met_li[11],met_li[12],met_li[13])
    ii += 1

def compute_ave_num_topics(cur, dbtable):
  # print "---------------------------------------------------"
  # print "   Precision     Recall       F-1 "
  # print "---------------------------------------------------"

  topic_count_li_li = []
  for item in range(1, 1001):
  # for item in range(1, 1501):
  # for item in range(1, 20):
    # get the topics from various columns
    query = "select m_39_topics_lem, m_43_topics_lem, 39_AND_43_lem, 39_OR_43_lem, opencalais_lem_split, nyt_manual_topics_lem from " + dbtable + " where id='" + str(item) + "'"
    cur.execute(query)
 
    # print "--------- Index: %d ---------" % item

    topic_groups = cur.fetchone()
    topic_count_li = []
    for each in topic_groups:
      topic_count_li.append(len(each.split(",")))
    topic_count_li_li.append(topic_count_li)
  
  for jj in topic_count_li_li:
    print "%d\t%d\t%d\t%d\t%d\t%d" % (jj[0],jj[1],jj[2],jj[3],jj[4],jj[5])

if __name__ == "__main__":
  # create a stop list -----------------------------//
  stop_li = open("stopwords.txt", "r").read().split()
  custom_stop_li = open("custom_stops.txt", "r").read().split()
  stop_list = list(set(stop_li + custom_stop_li))
  
  # connection to mysqldb --------------------------//
  dbuser = "username"
  dbpasswd = "password"
  hostname = "host.name.of.database"
  dbname = "nyt"
  dbtable = "nyt_1000"
  # dbtable = "nyt_3000"
  mysqldb = MySQLdb.connect(host=hostname, user=dbuser, passwd=dbpasswd, db=dbname)
  cur = mysqldb.cursor()

  reload(sys)
  sys.setdefaultencoding("utf-8")
  # ext_content(cur, dbtable)
  # clean_topics(cur, stop_list, dbtable)
  # split_underscore(cur, dbtable)
  # count_topics(cur, dbtable)
  # computeMetric(cur, dbtable, "p")
  # computeMetric(cur, dbtable, "r")
  computeMetric(cur, dbtable, "f1")
  # compute_ave_num_topics(cur, dbtable)