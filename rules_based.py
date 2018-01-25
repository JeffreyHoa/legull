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


import operator
from wordnet_systems import *
import highlight
import analyse_url_new
import sys


DEFAULT_SEPARATOR = "############# I'm default separator ##########"
separator_len = 100

	# Used to get the N most common pre-processed words for the Rule-Based Extractor.
def get_N_most_common_words(url, N, log=False):
	# Define dict for finding top N most common pre-processed words.
	processed_dict = {}

	#keep pre processed sentences for match_words_to_sentence function
	pp_sentences = list()

	if(log):
		print("#"*separator_len)
		print("N most common pre-processed words from document")
		print("#"*separator_len)

	print("\nurl:", url)

	# find the link number
	link_number = 0
	link_flag = 0
	with open("./output/Case_Links.txt") as f:

		for link in f:
			if url in link:
				link_flag = 1
				break
			else:
				link_number = link_number+1

	if(not(link_flag)):
		print("\nDocument not contained in Case_Links.txt.\nModule requires pre-processed data to continue.\n")
		print("\nPlease select a link from the Case_Links.txt file, and first run 'Train Bayesian Model Demo' from the legull UI.\n")
		return None, None

	if(log):
		print("\n\tLoad data from training_data_preprocessed.txt...")

	# find the correct preprocessed data.
	with open( "./output/train_data_preprocessed.txt") as file:
		separator_number = 0
		flag = 0
		pp_words = []
		for line in file:
			if line.startswith(DEFAULT_SEPARATOR):
				# if matches, then we have correct section.
				if flag == 1:
					break
				if link_number == separator_number:
					flag = 1
				else:
					# if it enters here, it has already found the correct document.
					if pp_words:
						break

					separator_number = separator_number + 1

			elif (flag == 1):
				# line in the form: 'tenant:14 tribunal:9 hearing:9 landlord:6 ... intimidated:1 \n'
				p_line = line.replace("\n", "")
				pp_sentences.append(line)
				words = p_line.split()
				#pp_sentences.append(words)
				for word in words:
					result = word.split(":")
					if result[0] in processed_dict:
						processed_dict[result[0]] += int(result[1])
					else:
						processed_dict[result[0]] = int(result[1])
				#pp_words.extend(words)

	'''
	pp_words currently in the form:
	['tenant:14 tribunal:9 hearing:9 landlord:6 ... intimidated:1 \n', 'application:3 ... \n']
	'''
	#print(pp_words)
	'''
	# needs result of preprocessing function as input arg:
	processed_dict = {}
	for i in range(1, len(result)):
	    word_stat_line = result[i][0]
	    for j in range(0, len(word_stat_line)):
	        # Get overall count of pre-processed words
	        if str(word_stat_line[j]['word']) in processed_dict:
	            processed_dict[str(word_stat_line[j]['word'])] += int(word_stat_line[j]['count'])
	        else:
	            processed_dict[str(word_stat_line[j]['word'])] = int(word_stat_line[j]['count'])
	'''

	# Move dict to list
	sorted_x = sorted(processed_dict.items(), key=operator.itemgetter(1), reverse=True)

	if(log):
		print("\n\tSort and get N most common pre-processed words in document...\n")
		print([x[0] for x in sorted_x[:N]])

	#Next:
	#Remove certain words such as tribunal which are very common.
	'''
	Improvements:
	1. Be able to read from train_data_proprocessed.txt. Done
	2. moved to after term_statistic in demo, as train_data_preprocessed.txt is present then. Done

	highlight.py can be used to highlight K sentences from an input list of sentence indexes
	Hence, next step is to return a list of sentence indexes where each word (remember
	it is pre-processed) occurs.

	Therefore, returned list (or output file) should be in the form of a list,
	where each element is a list with the first element being the word, and following elements
	being all the sentence numbers where that word occurs.

	Note that matching a word here is the same method as wordnet model, matching synonym to synonym.
	'''

	common_list = list()
	common_list = match_words_to_sentence(pp_sentences, sorted_x[:N], logPrint=False)
	'''
	print(common_list)

	highlight_list = list()

	highlight_list = highlight_sentence(common_list)

	print()
	print(highlight_list)
	print()
	highlight.hightlight_html(highlight_list, len(highlight_list))
	'''
	return [x[0] for x in sorted_x[:N]], common_list

'''
Also includes synonym matching for sentences that contain N most common words.
'''
def match_words_to_sentence(list_of_strings, N_most_common, logPrint=False):

	score_list = list()

	# list for sentence #'s that contain a word.
	#example for N = 5: [[1,0,0,0,1], [0,0,0,0,0], [0,1,1,0,1]]
	# sentence 1 has words 1 and 5, sentence 2 has nothing, sentence 3 has words 2,3,5.
	common_list = list()
	common_list_words = list()
	common_list_words = [x[0] for x in N_most_common]
	common_list_set = set(common_list_words)

	#cphrases = ""

	# return variables with form '(str)word1:(int)count (str)word2:(int)count'
	#(cphrases, list_of_strings) = get_keywords_summ(list_of_strings, cphrases, logPrint)

	sentences_synonyms = []

	for i in range(0, len(list_of_strings)):
		sentences_synonyms.append(get_synonyms_from_string(list_of_strings[i]))

	for strings_synonyms in sentences_synonyms:
		#for strings in list_of_strings: 					## if sentence synonyms not used
		common_list_sentence = ['0' for _ in N_most_common]
		for word in strings_synonyms:

			# separate ':(int)count' from each word
			word, num = word.split(':')

			# N most common functionality:
			if word in common_list_set:
				for common_word_index in range(len(common_list_words)):
					if word == common_list_words[common_word_index]:
						common_list_sentence[common_word_index]=word


		common_list.append(common_list_sentence)
			# common_list contents: For N = 5: [[appeal,0,0,0,tribunal], [0,0,0,0,0], [0,0,law,0,tribunal]]
	return common_list

def highlight_sentence(common_list, word_to_highlight):
	# Should make highlighter ignore words 'tenant', 'landlord', 'tribunal' (too frequent)
	highlight_list = list()

	# loop per sentence
	for sentence_num in range(len(common_list)):
		matched_word_list = common_list[sentence_num]

		#loop per word in N most common
		# from 4th (index 3) most common word:
		#for matched_word_index in range(4, len(common_list[0])):

		#for matched_word_list in common_list:
		for word_num in range(len(matched_word_list)):
			if (matched_word_list[word_num] == word_to_highlight):# and break_flag == 0):
				# if N most common words in sentence, add it to highlighting list.
				highlight_list.append(sentence_num)
				#break

	return highlight_list


def demo_rules_based(url='http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2014/3.html', N=10, word_num = 6):

	word_num = int(word_num)
	N = int(N)

	if(word_num > N):
		print("\nWord number can only be less than or equal to N.\n")
		return

	# need this to run highlighter
	analyse_url_new.get_details(url)

	#common_list = list()
	N_most_common, common_list = get_N_most_common_words(url, N, log=True)
	if(N_most_common == None):
		return

	highlight_list = list()
	#N_most_common is application in CATCD/2017/9
	'''
	Change the N_most_common[6] input with any word in the string list "N_most_common"
	from get_N_most_common_words.
	'''

	print("\nWord being tested:", N_most_common[word_num-1])

	highlight_list = highlight_sentence(common_list, N_most_common[word_num-1])

	print()
	print(highlight_list)
	print()
	highlight.hightlight_html(highlight_list, len(highlight_list)-1)

	#takes "url" and "# of most common words" as input
if(len(sys.argv) == 2):
	url = sys.argv[1]
	demo_rules_based(url)
elif(len(sys.argv) == 4):
	url = sys.argv[1]
	N = sys.argv[2]
	word_num = sys.argv[3]
	demo_rules_based(url, N, word_num)
else:
	demo_rules_based()
