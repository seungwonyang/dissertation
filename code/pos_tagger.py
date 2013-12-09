# pos_tagger.py
# This script is adopted from Jacob Perkins's "Part of Speech Tagging with NLTK"
# ULR: http://streamhacker.com/2008/11/03/part-of-speech-tagging-with-nltk-part-1/
# July 28, 2013

import itertools, nltk.tag
from nltk.corpus import brown, conll2000, treebank
from nltk import word_tokenize

class PosTagger:
	def __init__(self, brown, conll2000, treebank):		 
		self.conll_train = conll2000.tagged_sents('train.txt')
		self.conll_test = conll2000.tagged_sents('test.txt')	 
		treebank_cutoff = len(treebank.tagged_sents()) * 2 / 3
		self.treebank_train = treebank.tagged_sents()[:treebank_cutoff]
		self.treebank_test = treebank.tagged_sents()[treebank_cutoff:]

	def backoff_tagger(self, tagged_sents, tagger_classes, backoff=None):
		if not backoff:
			backoff = tagger_classes[0](tagged_sents)
			del tagger_classes[0]
		for cls in tagger_classes:
			tagger = cls(tagged_sents, backoff=backoff)
			backoff = tagger
		return backoff

	def train_raubt_tagger(self, train_sents):
		word_patterns = [
		    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
		    (r'.*ould$', 'MD'),
		    (r'.*ing$', 'VBG'),
		    (r'.*ed$', 'VBD'),
		    (r'.*ness$', 'NN'),
		    (r'.*ment$', 'NN'),
		    (r'.*ful$', 'JJ'),
		    (r'.*ious$', 'JJ'),
		    (r'.*ble$', 'JJ'),
		    (r'.*ic$', 'JJ'),
		    (r'.*ive$', 'JJ'),
		    (r'.*ic$', 'JJ'),
		    (r'.*est$', 'JJ'),
		    (r'^a$', 'PREP'),
		]
		raubt_tagger = self.backoff_tagger(train_sents, [nltk.tag.AffixTagger, nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
		    backoff=nltk.tag.RegexpTagger(word_patterns))
		return raubt_tagger

	def prt_head_corpus(self, corpus):
		print corpus[:5]

# ---- train with all treebank ----------------------------
pt = PosTagger(brown, conll2000, treebank)	

# train using the ENTIRE conll2000 corpus
raubt_trained = pt.train_raubt_tagger(conll2000.tagged_sents())