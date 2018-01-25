# Copyright 2017 Jeffrey Hoa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -*- coding: utf-8 -*-
'''
use:
from wordnet_systems import *
rank_sentences_mod_wordnet(['this is a trial string', 'Lets examine another sentence'], 'this is a rather empty string meant for test purposes.')

'''

import nltk
from nltk.corpus import wordnet as wn

import string #for punct
from myParser import Parser
from summarizer import Summarizer

#obsolete
# Takes a word string as input, returns list of strings as output.
def get_all_synonyms(arg1):
	strings = list()
	for synset in wn.synsets(arg1):
		for lemma in synset.lemmas():
			strings.append(lemma.name())
	return strings

# Takes a word string as input, returns unique list of strings as output.
def get_synonyms(arg1):
	strings = set()
	for synset in wn.synsets(arg1):
		for lemma in synset.lemmas():
			strings.add(lemma.name())
	return list(strings)

def split_string_to_list(my_string):
	cphrases_list = list()
	[cphrases_list.append(word.strip(string.punctuation)) for word in my_string.split()]	#keeps punctuation e.g won't]
	return cphrases_list

# Used for catchwords. Includes word weighting for synonyms, determined by original catchword word count.
def get_synonyms_from_string(my_string):
	#cphrases_list = list()
	cphrases_synonyms = list()
	#[cphrases_list.append(word.strip(string.punctuation)) for word in my_string.split()]	#keeps punctuation e.g won't
	cphrases_list = split_string_to_list(my_string)

	for cword in cphrases_list:

		# separate ':(int)count' from each word
		cword, num = cword.split(':')

		# This part will miss acronyms such as IT (lemmatise step removes either way)
		if len(cword) < 3:
			continue
		temp_set = set()
		# add original word
		temp_set.add(cword)
		# add synonyms, ignore duplicates
		temp_set.update(get_synonyms(cword))
		#include word count
		temp_set = [s + ':' + num for s in list(temp_set)]
		cphrases_synonyms.extend(temp_set)
		#cphrases_synonyms.extend(get_synonyms(cword))

	return cphrases_synonyms

# Modified old preprocessing function. sentences in list form.
def get_keywords_summ(list_sentences, cphrases, logPrint=False):

	## 1. Remove '\n','\t', etc and make it readable.
	for s in list_sentences:
		s = " ".join(s.replace("\n", " ").split())
	# Run through Summarizer
	summarizer = Summarizer()
	result = summarizer.summarize(list_sentences, cphrases)
	# above gets lists of {'word': 'decision', 'count': 1}, for each sentence it is given.
	# convert {'word': 'decision', 'count': 1} to form (str)word:(int)count for each word.
	'''
	if(logPrint):
		print("\tConvert pre-processed words to dictionary layout:")
		print("\t\t{ 'word':count }")
	'''
	cphrase_pp = ""
	sentences = ""
	list_sentences = list()

	for i in range(0, len(result)):
		sentences = ""

		word_stat_line = result[i][0]
		for j in range(0, len(word_stat_line)):
			if i >= 1:
				sentences = sentences + " " + str(word_stat_line[j]['word'])+":"+str(word_stat_line[j]['count'])
			else:
				cphrase_pp = cphrase_pp + " " + str(word_stat_line[j]['word'])+":"+str(word_stat_line[j]['count'])
		if i > 0:
			sentences.strip()
			list_sentences.append(sentences.lstrip())

	return (cphrase_pp.lstrip(), list_sentences)

'''
Gets synonyms for words in catchphrases, then compares to synonyms of strings list.
'''
def rank_sentences_mod_wordnet(list_lineInDoc, logPrint=False):

	score_list = list()

	cphrases = list_lineInDoc[0]
	list_of_strings = list_lineInDoc[1:]

	# return variables with form '(str)word1:(int)count (str)word2:(int)count'
	(cphrases, list_of_strings) = get_keywords_summ(list_of_strings, cphrases, logPrint)

	if(logPrint):
		print("\n\tGet Synonyms for each catch word to expand comparison set.")
		print("\tEach synonym inherits 'count' score of parent word.")
		print("\n\tReasons for this approach:")
		print("\t\t1. Consistent linear scoring if multiple of the same matching word in sentence.")
		print("\t\t2. Due to lemmatisation step in pre-processing, many words are in base forms.")
		print("\t\t   Hence, all synonym words treated equally in set expansion.")

	# Gets synonyms. Note we do not need to get keywords again as there are none in sentences list.
	cphrases_synonyms = get_synonyms_from_string(cphrases)

	if(logPrint):
		print("\n\tSynonyms also collected for each sentence in case law document body.")
		print("\n\tAdvantages of comparing catch word synonyms to sentence synonyms:")
		print("\t\t1. Increases chance to match words.")
		print("\t\t2. Decreases risk of tie scores for sentences, especially for small case document.")
		print("\n\tDisadvantages:")
		print("\t\t1. Excessive matching: There are a variable amount of synonyms provided")
		print("\t\t   by the Wordnet API for each word. Comparing synonyms to synonyms can")
		print("\t\t   overly boost sentence score beyond a more suitable sentence.")
		print("\t\t2. While the chance to match words better separates the rankings, only the")
		print("\t\t   top K sentences are relevant for this retrieval system.")

	sentences_synonyms = []

	for i in range(0, len(list_of_strings)):
		sentences_synonyms.append(get_synonyms_from_string(list_of_strings[i]))

	# add all synonyms to dictionary with count
	cwords_dict = {}
	for cword in cphrases_synonyms:
		cword, num = cword.split(':')
		if cword in cwords_dict:
			cwords_dict[cword] += int(num)
		else:
			cwords_dict[cword] = int(num)

	for strings_synonyms in sentences_synonyms:
	#for strings in list_of_strings: 					## if sentence synonyms not used

		count = 0
		for word in strings_synonyms:
		#string_list = split_string_to_list(strings) 	## if sentence synonyms not used
		#for word in string_list: 						## if sentence synonyms not used

			# separate ':(int)count' from each word
			word, num = word.split(':')

			if word in cwords_dict:
				count += int(num) * cwords_dict[word]

		score_list.append(count)

	return score_list


'''
Test code
'''

#list_scores = rank_sentences_mod_wordnet(['this is a test string with a purpose', 'Lets examine another sentence'], 'this is a rather empty string meant for test purposes. Examine super string theory.')
#print(list_scores)

#get_keywords('this is a rather empty string meant for test purposes.\n hello \n But here it is, I hope you like this sentence.')
#get_keywords_summ(['this is a rather empty hollow string meant for test purposes. test hello there this is also part of sentence 1.','this is a test trial string'], 'Lets examine another sentence as our catchphrase sentence for testing.')


