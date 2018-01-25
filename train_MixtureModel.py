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
This .py is for final demo and testing APIs as well.
Created in August 29th by Jeffrey
'''
import get_url
import analyse_url_new
import os
import numpy as np
import time
from summarizer import Summarizer
from myParser import Parser
#import highlight
import colored

from wordnet_systems import *


DEFAULT_SEPARATOR = "############# I'm default separator ##########"
separator_len = 100


def append(list_lineInDoc, start_idx, \
            train_data_path = "./output/train_data.txt", \
            separator = DEFAULT_SEPARATOR):

    f=open(train_data_path,'a')

    # Write separator.
    f.write(separator+str(start_idx))
    f.write('\n')

    for i in range(0, len(list_lineInDoc)):
        '''
            REASON TO CHANGE:
            when write sentences into file, some sentences contain some special
            charaters that cannot be encoded, which will raise an error
            like below:
            "UnicodeEncodeError: 'ascii' codec can't encode character '\u2013'
            in position 15: ordinal not in range(128)"

            In order to solve this issue, I added the code below, that will
            ignore those special characters. I think ignore those special
            characters won't affect our learning result.
        '''
        info_pre_encode = list_lineInDoc[i].encode("ascii", "ignore")
        info_to_write = info_pre_encode.decode('ascii')

        # Write catchphrases and sentences.
        f.write(info_to_write)
        f.write('\n')

    f.close()



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



def pre_processing(list_lineInDoc, caseIdx, output_path="./output/train_data_preprocessed.txt", separator = DEFAULT_SEPARATOR, write_valid = True, logPrint=False):
    '''
    To Jeff:
    Implement here.
    '''

    if (logPrint == True):
        print("")
        print("\tPre-processing law case [%d]" % caseIdx)

    if (caseIdx == 0 and logPrint == True):
        print("")
        print("\t\t1.Remove \\n, \\t")
        print("\t\t2.Remove Punctuations.")
        print("\t\t3.Split Words.")
        print("\t\t4.Remove StopWords.")
        print("\t\t5.Lemmatize Words.")

    ## 1. Remove '\n','\t', etc and make it readable.
    for idx in range(len(list_lineInDoc)):
        list_lineInDoc[idx] = " ".join(list_lineInDoc[idx].replace("\n", " ").split())


    ## 2. Analyse catchphrase and sentences.
    catchphrase = list_lineInDoc[0]
    sentences   = list_lineInDoc[1:]

    summarizer = Summarizer()
    result = summarizer.summarize(sentences, catchphrase)

    ## 3. Save result.
    if (write_valid == True):
        f=open(output_path,'a')
        f.write(separator+str(caseIdx))
        f.write('\n')

        for i in range(0, len(result)):

            word_stat_line = result[i][0]
            for j in range(0, len(word_stat_line)):
                f.write(str(word_stat_line[j]['word']))
                f.write(":")
                f.write(str(word_stat_line[j]['count']))
                f.write(" ")

            f.write('\n')
        f.close()

    return result


################################################################################


def links_filter(output_path = "./output/Case_Links.txt"):
    '''
    Extended:
    To make sure a better ml algorithm performance:
    1) Catchphrase should be more than 10 words.
    2) Detail should be more than 10 sentences.
    we only deal with these cases with friendly structure.
    '''
    default_link = 'http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/2.html'

    print("")
    print("#"*separator_len)
    print("Analyse links and extract Catchwords and Sentences.")
    print("#"*separator_len)
    print("")

    if True == os.path.exists(output_path):
        print("Case_Links.txt exists, load links as following:\n")
    else:
        '''
        This .py is mainly for demo, so here
        we will show how to get 5-10 valid links and save them in Case_Links.txt.

        Implement here.
        '''
        #f=open(output_path,'w+')
        #f.write(default_link)
        #close(f)

        get_url.get_links_demo_scope(output_path)
        print("Create new Case_Links.txt.")


################################################################################


def read_links():
    '''
    Implement here.
    '''
    file = open( "./output/Case_Links.txt")
    links = []
    for i in file:
        links.append(i)

    return links


def catchphrase_and_detail_extractor(link):
    '''
    Analyse link and get catchphrase & sentences.

    Implement here.
    '''

    '''
    list: doc --> doc --> doc
           |
           |-[0] catchphrase
           |-[1] sentence 1 in detail
           |-[2] sentence 2 in detail
           |- ...
           |-[N] sentence N in detail

    list_lineInDoc=['STRATA APPEAL â€“ admission of new evidence â€“ works to be undertaken by Owners Corporation, maintenance or new works â€“ amendment of motions.',
              'The property known as [***] Avenue, Neutral Bay (SP 30995) comprises some 8 units with a total of 12 balconies. There are 4 units at the rear of the property being units 5, 6, 7 and 8. There are 6 balconies on the western side at the rear of the building and another 6 on the front and sides of the building. The balustrades along the western side are of glass and aluminium construction and they face west and offer views. The balustrades on the other sides (being 6 in total) are of masonry/brick construction.',
              'The appellants, who were also the applicants for the adjudication which includes balconies which were the subject of the resolution purportedly passed on 31 March 20215.']
    '''
    analyse_url_new.get_details(link)
    catchphrase = analyse_url_new.get_catchwords()
    list_lineInDoc = [catchphrase] + analyse_url_new.get_sentences()
    return list_lineInDoc



def link_analyser(input_path = "./output/Case_Links.txt"):

    # list_links:  [<link 1>, <link 2>, <link 3>, ... <link N>]
    list_links = read_links()
    idx = 0

    for link in list_links:
        print("\tLaw case [%d]: %s" % (idx, link))
        list_lineInDoc = catchphrase_and_detail_extractor(link)
        #print(list_lineInDoc)
        append(list_lineInDoc, idx)
        idx=idx+1

    print("Save <Catchwords> and <Sentences> in train_data.txt")

def term_statistic(input_path = "./output/train_data.txt", \
                   word_index_file = "./output/word_index.npz"):

    print("")
    print("#"*separator_len)
    print("Pre-process and create train data in train_data.npz")
    print("#"*separator_len)
    print("")
    print("Load <Catchwords> and <Sentences> from train_data.txt")
    pos = 0
    caseIdx = 0

    catchword_list = []
    bodyword_list  = []

    while(True):
        list_lineInDoc, pos = readOneCase(pos, train_data_path=input_path)
        #print(list_lineInDoc)
        #print(pos)

        if len(list_lineInDoc) == 0:
            break

        result = pre_processing(list_lineInDoc, caseIdx, logPrint=True);
        #print("pre_processing: ",result)

        '''
        [
        ([{'count': 1, 'word': 'appeal'},        {'count': 1, 'word': 'apple'}, {'count': 1, 'word': 'banana'}, {'count': 1, 'word': 'stratum'}], 4),
        ([{'count': 1, 'word': 'sent01apples'},  {'count': 1, 'word': 'sent01apple'}], 2),
        ([{'count': 1, 'word': 'sent02bananas'}, {'count': 1, 'word': 'sent02banana'}], 2)
        ]
        '''
        for i in range(0, len(result)):
            if (i == 0):
                word_stat_line = result[i][0]
                for j in range(0, len(word_stat_line)):
                    catchword_list.append(str(word_stat_line[j]['word']))
            else:
                word_stat_line = result[i][0]
                for j in range(0, len(word_stat_line)):
                    bodyword_list.append(str(word_stat_line[j]['word']))


        # make terms unique in set.
        catchword_list = list(set(catchword_list))
        bodyword_list  = list(set(bodyword_list))

        caseIdx += 1

    print("\nSave result in train_data_preprocessed.txt")


    # Save catchword_list and bodyword_list here.
    np.savez(word_index_file, catchword_list, bodyword_list)

    print("")
    print("\tCatchword dictionary:")
    for idx in range(0,2):
        print("\t\t%d: %s" % (idx, catchword_list[idx]))
    print("\t\t...")
    print("")
    print("\tBodyword  dictionary:")
    for idx in range(0,2):
        print("\t\t%d: %s" % (idx, bodyword_list[idx]))
    print("\t\t...")
    print("")

    print("Save Term Dictionary in word_index.npz")


    # build train Data for training later.
    build_trainData(catchword_list, bodyword_list)



def build_trainData(catchword_index, bodyword_index, input_path = "./output/train_data.txt"):
    print("\nBuild Train Data in One-hot encoding:\n")
    pos = 0
    caseIdx = 0

    # Get term set in catchphrase
    #print("catchword len:", len(catchword_index))
    # Get term set in body sentences
    #print("bodyword  len:", len(bodyword_index))


    #x_train = np.zeros((0, len(bodyword_index)))
    #y_train = np.zeros((0, len(catchword_index)))
    x_train_list = []
    y_train_list = []

    while(True):
        list_lineInDoc, pos = readOneCase(pos, train_data_path=input_path)
        #print(list_lineInDoc)
        #print(pos)

        if len(list_lineInDoc) == 0:
            break

        #print("\npre_processing: ing... ")
        result = pre_processing(list_lineInDoc, caseIdx, write_valid = False);
        #print("pre_processing: ",result)

        catchword_list = []
        bodyword_list  = []

        '''
        [
        ([{'count': 1, 'word': 'appeal'},        {'count': 1, 'word': 'apple'}, {'count': 1, 'word': 'banana'}, {'count': 1, 'word': 'stratum'}], 4),
        ([{'count': 1, 'word': 'sent01apples'},  {'count': 1, 'word': 'sent01apple'}], 2),
        ([{'count': 1, 'word': 'sent02bananas'}, {'count': 1, 'word': 'sent02banana'}], 2)
        ]
        '''
        for i in range(0, len(result)):
            if (i == 0):
                word_stat_line = result[i][0]
                for j in range(0, len(word_stat_line)):
                    catchword_list.append(str(word_stat_line[j]['word']))
            else:
                word_stat_line = result[i][0]
                for j in range(0, len(word_stat_line)):
                    bodyword_list.append(str(word_stat_line[j]['word']))

        # make terms unique in set.
        catchword_list = list(set(catchword_list))
        bodyword_list  = list(set(bodyword_list))


        #####################################
        # Create one sample for x_train data.
        #####################################
        x_train_sample = np.zeros(len(bodyword_index))
        for word in bodyword_list:
            idx = bodyword_index.index(word)
            x_train_sample[idx] = 1


        for word in catchword_list:
            #####################################
            # Create one sample for y_train data.
            #####################################
            y_train_sample = np.zeros(len(catchword_index))
            idx = catchword_index.index(word)
            y_train_sample[idx] = 1

            #print("One Sample:");
            #print(y_train_sample)
            #print(x_train_sample)

            '''
            Too slow!
            change from array to list.
            '''
            #y_train = np.row_stack((y_train, y_train_sample))
            #x_train = np.row_stack((x_train, x_train_sample))

            y_train_list.append(y_train_sample.tolist())
            x_train_list.append(x_train_sample.tolist())

        caseIdx += 1

    y_train = np.asarray(y_train_list)
    x_train = np.asarray(x_train_list)

    print("-"*separator_len)
    print("y_train:");
    print(y_train)
    print("  ");
    print("x_train:");
    print(x_train)
    print("-"*separator_len)
    print("  ");

    outfile = "./output/train_data.npz"
    np.savez(outfile, x_train, y_train)

    #npzfile = np.load(outfile)
    #print(npzfile.files)
    #print(npzfile['arr_0'])
    #print(npzfile['arr_1'])

    print("Save train data in train_data.npz\n")



def naive_bayes_posterior_mean(x, y, alpha=1, beta=1):
    """
    Given an array of features `x`,
    an array of labels `y`,
    class prior Dirichlet parameter `alpha`, and
    common class-conditional feature expectation `beta`
    return

    a posterior mean, `pi`, of `alpha` and
    a posterior mean, `theta` of the `beta`.

    NB: this is not the same as returning the parameters of the full posterior,
    but it is sufficient to calculate the posterior predictive density.
    """
    n_class = y.shape[1]
    n_feat = x.shape[1]

    # as a convenience, we allow both alpha and beta to be scalar values
    # which will be upcast to arrays for the common case of using priors to smooth
    # this is a no-op for alpha
    # but for beta, we must be explicit
    beta = np.ones(2) * beta

    pi_counts = np.sum(y, axis=0) + alpha
    pi = pi_counts/np.sum(pi_counts)

    theta = np.zeros((n_feat, n_class))

    for cls in range(n_class):
        docs_in_class = (y[:, cls]==1)
        class_feat_count = x[docs_in_class, :].sum(axis=0)
        theta[:, cls] = (class_feat_count + beta[1])/(docs_in_class.sum() + beta.sum())

    return pi, theta



def trainBayesianModel(input_path = "./output/train_data.npz"):

    print("#"*separator_len)
    print("Bayesian Model Training")
    print("#"*separator_len)

    npzfile = np.load(input_path)
    print("\nLoad train data from train_data.npz\n")

    #print(npzfile.files)
    xtrain = npzfile['arr_0']
    ytrain = npzfile['arr_1']

    pi_bar, theta_bar = naive_bayes_posterior_mean(xtrain, ytrain)

    link = "https://www.dropbox.com/s/pojefqxfkrr049w/demo.pdf?dl=0"
    print("Goto:", link)
    print("")
    print("-"*separator_len)
    print("pi:", pi_bar.shape)
    print(pi_bar)
    print("")
    print("theta:", theta_bar.shape)
    print(theta_bar)
    print("-"*separator_len)

    outfile = "./output/train_model.npz"
    np.savez(outfile, pi_bar, theta_bar)

    #npzfile = np.load(outfile)
    #print(npzfile.files)
    #print(npzfile['arr_0'])
    #print(npzfile['arr_1'])

    print("\nSave model in train_model.npz\n")



def predictBayesianModel(sentenceList = ['hello world occupation lease', 'machine learning board', 'machine learning lease occupation'],
                         input_path = "./output/train_model.npz",
                         word_index_file = "./output/word_index.npz", logPrint = False):

    print("Load model:      train_model.npz")
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

    print("Load dictionary: word_index.npz")
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

    link = "https://www.dropbox.com/s/pojefqxfkrr049w/demo.pdf?dl=0"
    print("\nGoto:", link)

    # NB: sentence ith, from 1 to end.
    return scoreRecord


def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;


def retTopK(url='http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/9.html', log = False):

    # Read the first case in train_data.txt for testing.
    #input_path = "./model/train_data.txt"
    #list_lineInDoc, pos = readOneCase(file_pos, train_data_path=input_path)

    '''
    get catch words and sentences of a url.
    '''
    #if(log): print('\nRetrieving case sentences...')

    #analyse_url_new.get_details(url)
    catchphrase = analyse_url_new.get_catchwords()
    sentences   = analyse_url_new.get_sentences()

    ## Remove '\n','\t', etc and make it readable.
    catchphrase = " ".join(catchphrase.replace("\n", " ").split())
    for idx in range(len(sentences)):
        sentences[idx] = " ".join(sentences[idx].replace("\n", " ").split())


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

    print("#"*separator_len)
    print("Bayesian Model Prediction")
    print("#"*separator_len)
    print("")
    print("New law case:", url)
    print("")
    print("Extract catchwords and sentences.")

    print("Pre-processing law case.")
    print("Pre-processing catchwords and sentences.")
    print("")
    s1 = predictBayesianModel(list_lineInDoc, logPrint=log)

    print("\nScore list for each sentence:")
    print([ float("%.2f" % elem) for elem in s1 ])
    print("")

    print("#"*separator_len)
    print("Wordnet Model Prediction")
    print("#"*separator_len)
    print("")
    print("New law case:", url)
    print("")
    print("Extract catchwords and sentences.")

    print("Pre-processing law case.")
    print("Pre-processing catchwords and sentences.")
    print("")
    s2 = rank_sentences_mod_wordnet(list_lineInDoc, logPrint=log)

    print("\nScore list for each sentence:")
    print([ float("%.2f" % elem) for elem in s2 ])
    print("")

    print("")
    print("#"*separator_len)
    print("Mixture Model: alpha * Bayesian Model + (1-alpha) * Wordnet Model")
    print("#"*separator_len)

    # Normalization by Jeff.
    s1_norm = []
    s2_norm = []

    for score in s1:
        s1_norm.append( MaxMinNormalization(score, max(s1), min(s1)) )
    for score in s2:
        s2_norm.append( MaxMinNormalization(score, max(s2), min(s2)) )


    # weighting on ML. Below 0.5 due to inaccuracy of method.
    if(log): print("\nFunction to return a ranked list of the most relevant sentences.\n")
    if(log):
        print("\tAlpha = 0.25")
        print("\tBayesian Model given less weighting due to its relative inaccuracy.")
        print()
    alpha = 0.25


    if(log):
        print("************************************************")
        print("Normalised Scores for Bayesian and Wordnet Model")
        print("************************************************")
        print('\nBayesian Model Result Normalised:')
        print([ float("%.2f" % elem) for elem in s1_norm ])
        print('\nWordnet Model Result Normalied:')
        print([ float("%.2f" % elem) for elem in s2_norm ])

    combined = list()
    for i in range(0, len(s1_norm)):
        # Use alpha weighting
        combined.append(s1_norm[i]*alpha + s2_norm[i]*(1-alpha))
    combined_np = np.array(combined)
    combined_sorted = combined_np.argsort()[::-1].argsort()
    if(log):
        print("\nRanks of Combined Bayesian and Wordnet Model (From sentence 1 to end):")
        print([int(1+elem) for elem in list(combined_sorted)])
        print()
    '''
    Previous Method: Combine by ranks instead of scores.

    # Sort the ranks for sentences so that array stays ordered by sentence #,
    # and array populated by rank of sentence #
    s1_np = np.array(s1)
    s1_sorted = s1_np.argsort()[::-1].argsort()
    s2_np = np.array(s2)
    s2_sorted = s2_np.argsort()[::-1].argsort()
    if(log):
        print("\nRanks of Bayesian Model (first # is sentence 0 etc):")
        print (list(s1_sorted))
        print("\nRanks of Wordnet Model (first # is sentence 0 etc):")
        print (list(s2_sorted))

    combined = list()
    for i in range(0, len(s1_sorted)):
        # Use alpha weighting
        combined.append(s1_sorted[i]*alpha + s2_sorted[i]*(1-alpha))    # <----- double check please.

    # Sort the combined ranks for sentences so that array stays ordered by sentence #,
    # and array populated by rank of sentence #
    combined_np = np.array(combined)
    combined_sorted = combined_np.argsort().argsort()
    if(log):
        print("\nFinal ranks - (first # is sentence 0 etc):")
        print (list(combined_sorted))
    #print([ "%d" % elem for elem in list(combined_sorted) ])
    #use enumerate
    '''
    # Finally get ranks from top to least, with sentence # in the output array.
    combined_rank_ordered = combined_sorted.argsort()
    if(log):
        print("************************************************")
        print("Final ranks (sentence # shown):")
        print("************************************************")
        print([int(1+elem) for elem in list(combined_rank_ordered)])
        #print (list(combined_rank_ordered)[:topK])
        print()

    return list(combined_rank_ordered)

'''
How to use:
1) Run retTopK and keep list of top K sentence indexes
2) Run labelTopK with K, above list, and url.
'''
def labelTopK(topK_sentence_indexes, url, topK = 10):

    analyse_url_new.get_details(url)
    sentence_list = analyse_url_new.get_sentences()

    TopKSentences = []
    for i in range(0, topK):
        TopKSentences.append(sentence_list[topK_sentence_indexes[i]])

    return TopKSentences





################################################################################
# Start from here
################################################################################

def demo_training_bayesianModel():
    '''
    Delete these files, we will create a new one.
    '''
    train_data_path = "./output/train_data.txt"
    if True == os.path.exists(train_data_path):
        os.remove(train_data_path)
    train_data_processed_path = "./output/train_data_preprocessed.txt"
    if True == os.path.exists(train_data_processed_path):
        os.remove(train_data_processed_path)

    '''
    Get several links and save them into Case_Links.txt
    '''
    links_filter()

    '''
    Extract catchphrase and detail part and save them into train_data.txt
    for nlp pre-processing in the next stage.
    '''
    link_analyser()

    '''
    Load law cases from train_data.txt.
    Save its statistical info in train_data_preprocessed.txt
    Build xtrain and ytrain.
    Save in ./output/train_data.npz
    '''
    term_statistic()

    '''
    Train to get Bayesian model.
    Save in ./output/train_model.npz
    '''
    trainBayesianModel()

    '''
    Predict through Bayesian model.
    Predict through Wordnet model.
    Combine ranks of sentences by each approach.
    Print out rank of all sentences (sentence # shown)
    '''
    url = 'http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/9.html'

    index_topK = retTopK(url, log=True)

    '''
    Pop out case webpage with highlighted sentences.
    '''
    #labelTopK(index_topK, url, 5)
    K = 5
    print("Top", K, "sentences:")
    print([int(1+elem) for elem in index_topK[:K]])
    print()
    #highlight.hightlight_html(index_topK, K)


# Let's run it.
demo_training_bayesianModel()




