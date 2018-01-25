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


'''
This .py is for demo and testing APIs as well.
Created in Sep 24th by Jie HAO (Jeffrey)
'''
import analyse_url_new
import os
import numpy as np
import pylab as pl
from summarizer import Summarizer
from myParser import Parser
from wordnet_systems import *
import highlight
import colored
DEFAULT_SEPARATOR = "############# I'm default separator ##########"



'''
arg1:
    start_idx for file.seek to the start point of the case required.
arg2:
    train data (train_data.txt) path.
arg3:
    separator between cases --> "############"
'''
def readOneCase(seek_pos,
                train_data_path,
                separator = DEFAULT_SEPARATOR):
    list_lineInDoc = []

    with open(train_data_path) as f:
        # start from the first char in one line.
        f.seek(seek_pos)

        while f.tell() < os.fstat(f.fileno()).st_size:
            line = f.readline()
#            print(line)

            if line.startswith(separator):
#                print("start with.")

                while f.tell() < os.fstat(f.fileno()).st_size:
                    case_line = f.readline()
#                    print(case_line)

                    if case_line.startswith(separator):
#                        print("DONE...")
                        return list_lineInDoc, f.tell()-len(case_line)

                    else:
#                        print("Append "+case_line)
                        list_lineInDoc.append(case_line)
#        print("EOF...")
        return list_lineInDoc, f.tell()

'''
function to find file position of a case document's training data (catch words and sentences)
in the train_data.txt file.
'''
def get_pos_from_train_data(website, separator = DEFAULT_SEPARATOR):
    site_filepos = 0
    with open('./output/Case_Links.txt', 'r') as f:
        for line in f:
            site_filepos = site_filepos + 1
            if website in line:
                break


        '''
        # if website is a number (e.g. case_doc #0)
        case_num = 0
        for line in f:
            if case_num == website:
                print(line)
            else:
                case_num = case_num + 1
        '''
        '''
        while True:
            line = f.readline()
            if not line: break
            if website in line:
                site_filepos = f.tell()- len(line)
                break

            else:
                print("how are you.")
        '''

    file_pos = 0
    with open('./output/train_data.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            if line.startswith(separator):
                site_filepos = site_filepos - 1
                if site_filepos == 0:
                    file_pos = f.tell()-len(line)

    return file_pos


def predictBayesianModel(sentenceList = ['hello world occupation lease', 'machine learning board', 'machine learning lease occupation'],
                         input_path = "./model/train_model.npz",
                         word_index_file = "./model/word_index.npz"):
    #print("\n-----------------------------------------")
    #print("Load model1: ./model/train_model.npz")
    #print("-----------------------------------------")
    #--------------------------------------------------
    npzfile = np.load(input_path)

    pi_bar = npzfile['arr_0']
    theta_bar = npzfile['arr_1']
    #print("\n[[pi]]:")
    #print(pi_bar)
    #print(pi_bar.shape)
    #print("\n[[theta]]:")
    #print(theta_bar)
    #print(theta_bar.shape)

    #print("\n-----------------------------------------")
    #print("Load model2: ./model/word_index.npz")
    #print("-----------------------------------------")
    #--------------------------------------------------
    npzfile2 = np.load(word_index_file)

    catchword_index = npzfile2['arr_0']
    bodyword_index  = npzfile2['arr_1']
    #print("\n[[catchword index]]:")
    #print(catchword_index)
    #print(catchword_index.shape)
    #print("\n[[bodyword index]]:")
    #print(bodyword_index)
    #print(bodyword_index.shape)
    #--------------------------------------------------

    scoreRecord = []
    parser = Parser()
    catchword_list = catchword_index.tolist()
    bodyword_list  = bodyword_index.tolist()

    #--------------------------------------------------
    # Get catchword_positionList
    #--------------------------------------------------
    catchwords = sentenceList[0]
    (keywords, wordCount) = parser.getKeywords(catchwords)

    catchword_positionList = []
    #print("keywords: ", keywords)
    for elem in keywords:
        word  = elem['word']
        count = elem['count']

        idx = catchword_list.index(word) if word in catchword_list else -1
        if (idx != -1):
            #print("appending ", catchword_list[idx])
            catchword_positionList.append(idx)

    #Debug
    #print("catchword_positionList:", catchword_positionList)
    #for catchwordPos in catchword_positionList:
        #print(catchword_list[catchwordPos])

    #--------------------------------------------------
    # Calculate score for each word in body sentence.
    # The first sentence is catchphrases.
    #--------------------------------------------------
    for idx in range(1, len(sentenceList)):
        (keywords, wordCount) = parser.getKeywords(sentenceList[idx])

        sentence_score = 0
        '''
        1) get the position list of catch words in predicted case.
        2) for each word in each sentence, find the scores for each catchword in theta_bar.
        3) add these scores which will be the final for one word in this sentence.
        4) evaluate next word... until the end of this sentence.
        5) goto 2).
        '''

        ## print("----------- sentence --------------")
        for elem in keywords:
            # Jeff: For each word in body sentence.
            word  = elem['word']
            count = elem['count']
            ## print("sentence word      :", word)
            ## print("sentence word count:", count)
            ## print(" ")

            word_score = 0;

            wordInSentence_idx = bodyword_list.index(word) if word in bodyword_list else -1
            if (wordInSentence_idx != -1):

                # Jeff: For each word in catchphrase
                for catchwordIdx in catchword_positionList:
                    ## print("* theta_bar[",idx, "][", catchword_list[catchwordIdx], "]")
                    ## print("* score:", theta_bar[idx][catchwordIdx])
                    ## print(" ")

                    word_score += theta_bar[idx][catchwordIdx]

            sentence_score += word_score*count


        scoreRecord.append(sentence_score)

    # NB: sentence ith, from 1 to end.
    #print("\nScore list for each sentence:")
    #print([ float("%.2f" % elem) for elem in scoreRecord ])
    #print("")

    return scoreRecord




def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;



'''
    Predict through Bayesian model.
    Predict through Wordnet model.
    Combine ranks of sentences by each approach.
    Print out top K Sentences (sentence #)
'''
def retTopK(url='http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/3.html', log = False, showGraph = False):
    # weighting on ML. Below 0.5 due to inaccuracy of method.
    if(log): print("\nFunction to return the top K (K is defined by the user) most relevant sentences.\n")

    if(log): print("An alpha of 0.25 is used to give less weighting to the Bayesian Model due to its relative inaccuracy.")
    alpha = 0.25

    # Read the first case in train_data.txt for testing.
    #input_path = "./model/train_data.txt"
    #list_lineInDoc, pos = readOneCase(file_pos, train_data_path=input_path)

    '''
    get catch words and sentences of a url.
    '''
    if(log): print('\nRetrieving case sentences...')

    if (url):
        analyse_url_new.get_details(url)

    catchphrase = analyse_url_new.get_catchwords()
    sentences = analyse_url_new.get_sentences()
    list_lineInDoc = []
    list_lineInDoc.append(catchphrase)
    list_lineInDoc.extend(sentences)
    '''
    list_lineInDoc is a sentence list..
    list_lineInDoc[0] is catchphrase.
    list_lineInDoc[1:n] is body sentences.
    '''
    s1 = []
    s2 = []

    print("************************************************")
    print("Bayesian Model Prediction")
    print("************************************************")
    print("")
    s1 = predictBayesianModel(list_lineInDoc)

    #if(log): print('\nRunning Wordnet Model...')
    print("************************************************")
    print("Wordnet Model Prediction")
    print("************************************************")
    print("")
    s2 = rank_sentences_mod_wordnet(list_lineInDoc)

    # Normalization by Jeff.
    s1_norm = []
    s2_norm = []

    for score in s1:
        s1_norm.append( MaxMinNormalization(score, max(s1), min(s1)) )
    for score in s2:
        s2_norm.append( MaxMinNormalization(score, max(s2), min(s2)) )

    print("************************************************")
    print("Scores for the Bayesian Model and Wordnet Model")
    print("************************************************")

    print('\nBayesian Model Score List:')
    print([ float("%.2f" % elem) for elem in s1 ])
    print('\nWordnet Model Score List:')
    print([ float("%.2f" % elem) for elem in s2 ])

    if(log):
        print('\nBayesian Model Normalised Score List:')
        print([ float("%.2f" % elem) for elem in s1_norm ])
        print('\nWordnet Model Normalied Score List:')
        print([ float("%.2f" % elem) for elem in s2_norm ])

    # combine scores of the two methods, get rank of combined scores.
    combined = list()
    for i in range(0, len(s1_norm)):
        # Use alpha weighting
        combined.append(s1_norm[i]*alpha + s2_norm[i]*(1-alpha))
    combined_np = np.array(combined)
    combined_sorted = combined_np.argsort()[::-1].argsort()
    if(log):
        print("\nRanks of Combined Bayesian and Wordnet Model (first # is sentence 0 etc):")
        print ([int(1+elem) for elem in list(combined_sorted)])

    # Finally get ranks from top to least, with sentence # in the output array.
    combined_rank_ordered = combined_sorted.argsort()

    print("\nFinal sentence rank (sentence # shown): [more important --> less important]")
    print([int(1+elem) for elem in list(combined_rank_ordered)])
    print()

    ################################################################
    # Jeff: this is crazy.
    ################################################################
    final_score = np.zeros(len(s1_norm))
    for idx in range(len(s1_norm)):
        final_score[idx] = s1_norm[idx]*alpha+s2_norm[idx]*(1-alpha)

    x = np.arange(1, len(final_score)+1)
    y  = final_score

    x1 = np.arange(1, len(s1_norm)+1)
    y1 = s1_norm

    x2 = np.arange(1, len(s2_norm)+1)
    y2 = s2_norm

    if (showGraph):
        pl.plot(x, y, 'r', x1, y1, 'b', x2, y2, 'g')

        pl.xlabel('Sentence')
        pl.ylabel('Normalised Score')
        pl.title('Red = Final Score, Blue = Bayseian, Green = Wordnet')
        pl.show(block=False)
    return list(combined_rank_ordered)

'''
How to use:
1) Run retTopK and keep list of top K sentence indexes
2) Run labelTopK with K, above list, and url.
'''
def labelTopK(topK_sentence_indexes, url, topK = 10):

    analyse_url_new.get_details(url, False)
    sentence_list = analyse_url_new.get_sentences()

    TopKSentences = []
    for i in range(0, topK):
        TopKSentences.append(sentence_list[topK_sentence_indexes[i]])

    return TopKSentences



################################################################################
# Start from here
################################################################################


# sentences assumed to start from sentence #0

#color_index = [i for i in range(19, 230)]

#url = 'http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATAP/2015/119.html'
#my_topK = retTopK(url)

#mss = "\nFinal sentence rank is: [more important --> less important]"
#color.print_bold(mss, color_index[0])
#del color_index[0]

#K = 5
#sentence_topK = labelTopK(my_topK, url, K)
#print(sentence_topK)
#print("Top", K, "sentences:")
#print([int(1+elem) for elem in my_topK[:K]])
#print()
#highlight.hightlight_html(my_topK, K)





