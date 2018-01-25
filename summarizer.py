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


#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Sat Sep 09 10:39:43 2017

@author: HAO
'''

from myParser import Parser

class Summarizer:
    def __init__(self):
        self.parser = Parser()

    '''
    test : it is catchphrase, one sentence.
    title: sentence list.
    '''
    def summarize(self, text, title):
        sentences = text
        result = []

        ## step 1, get term list of catchphrase.
        (catchphrase_keywords, catchphrase_wordCount) = self.parser.getKeywords(title)
        result.append( (catchphrase_keywords, catchphrase_wordCount) )

        catchword_list = [catchphrase_keywords[idx]['word'] for idx in range(len(catchphrase_keywords))]
        #print("[*catchword_list*]",catchword_list)

        ## step 2, get top k word list in sentences.
        ## 2.1 get term list of detail.
        #text_merged = " ".join(sentences)
        #(detail_keywords, detail_wordCount) = self.parser.getKeywords(text_merged)

        for idx in range(len(text)):
            (sentence_keywords, sentence_wordCount) = self.parser.getKeywords(text[idx])
            result.append( (sentence_keywords, sentence_wordCount) )

            word_list = [sentence_keywords[idx]['word'] for idx in range(len(sentence_keywords))]
            #print("\n[*word_list*]", word_list)


        return result



