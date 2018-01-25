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


# !/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Sat Sep 09 10:29:42 2017

@author: HAO (Jeffrey)
'''

import nltk.data
from nltk.stem import WordNetLemmatizer

import os

class Parser:
    def __init__(self):
        self.stopWords = self.getStopWords()
        '''
        Jeff: Set this path based on your local env.
        '''
        nltk.data.path.append('/usr/local/nltk_data/')

    def getKeywords(self, text):
        text  = self.removePunctations(text)
        words = self.splitWords(text)
        words = self.removeStopWords(words)
        words = self.lemmatizeWords(words)
        uniqueWords = list(set(words))

        keywords = [{'word': word, 'count': words.count(word)} for word in uniqueWords]
        keywords = sorted(keywords, key=lambda x: -x['count'])

        return (keywords, len(words))


    def splitWords(self, sentence):
        return sentence.lower().split()

    def removePunctations(self, text):
        return ''.join(t for t in text if t.isalnum() or t == ' ')

    def removeStopWords(self, words):
        return [word for word in words if word not in self.stopWords]

    def getStopWords(self):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/output/stopWords.txt') as file:
            words = file.readlines()

        return [word.replace('\n', '') for word in words]

    def lemmatizeWords(self, text):
        wordnet_lemmatizer = WordNetLemmatizer()
        return [wordnet_lemmatizer.lemmatize(word) for word in text]

