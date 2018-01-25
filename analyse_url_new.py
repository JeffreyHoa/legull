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
"""
Created on Thu Sep  7 13:10:18 2017

@author: Jeff

use a new method to grab content from a given URL

Input: URL
Output: a list which contains the catchwords and sentences from the body
"""


import urllib.request
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import sys
import re





'''
    analyse_text fuction takes a URL as argument
    
    Functionalities:
    (1) from the given URL, fetch the catchphrase(catchwords),
        store as a string.
    (2) get the general infomation from the given URL, 
        store in general_info for future use.
    (3) grab the main body part of the document, 
        split the whole body part into many "sentences", 
        stored in a list ---->> sentences
    (4) return a list --->>  list_doc = [catchphrase, sentences1, sen2, ...]
        
'''


def get_details(URL, UI=False):
    if(UI):
        print("************************************************")
        print("case link: " + URL)
        print("************************************************\n")

    global general_info, catchwords, sentences, soup
    general_info = []
    catchwords = ''
    body = ''
    plaintiff = ''
    defendent = ''
    title = ''
    decision = ''
    sentences = []
    
    try:    
        req = Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    #    page = urllib.request.urlopen(URL)
        html_content = urlopen(req).read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for td in soup.find_all('td'):
            general_info.append(td.text)
            
        for i in range(len(general_info)):
            if 'catchword' in general_info[i].lower() or 'catchphrase' in general_info[i].lower(): 
                if i+1 >= len(general_info):
                    print('################ bug appears at position 1.... ################')
                    sys.exit(0)
                catchwords = general_info[i+1]
                break
        catchwords = catchwords.replace('\n', '')
        
        num_useless = general_info.count('\n')
        for _ in range(num_useless):
            general_info.remove('\n')
        
    
        body_info = []
        num = 1
        
        while(True):
            info = soup.find_all('li', attrs = {'value': num})
            if len(info) > 1:
                info = [info[0]]
    #            print("################ bug appears at position 2.... ################")
            if len(info) == 0:
                # "Body infomation finished...."
                break
            else:
                next_quote = info[0].find_next("blockquote")
                if next_quote:
                    next_info = soup.find_all('li', attrs = {'value': num+1})
                    if not next_info:
                        body_info.append(info[0].text + next_quote.text)
                    else:
                        next_quote_copy = next_info[0].find_next("blockquote")
                        if next_quote == next_quote_copy:
                            body_info.append(info[0].text)
                        else:
                            body_info.append(info[0].text + next_quote.text)
                else:
                    body_info.append(info[0].text)
            num += 1
    
        for item in body_info:
            sentences.append(make_sentences(item))
            
    #    return [catchwords] + sentences
        return True
    except HTTPError:
        print(HTTPError)
        print("Opps! The page is not responding...")
        sys.exit(1)
    
    
    
'''
    By giving a mess sentence which has many newline between a single line or
    much space between two words.
    Convert to a normal human readable sentence and return it.
'''
    
def make_sentences(mess):
    mess = mess.replace('\t', ' ')
    temp = mess.split('\n')
    good_line = ' '.join(part for part in temp)
    return good_line


'''
    input which part of general info that we need. And the function will return
    the information to the user.
'''

def get_general_info(case_part):
    pos = 0
    global general_info
    for i in range(len(general_info)):
        if case_part == "title":
            if "case name" in general_info[i].lower() or "case title" in general_info[i].lower():
                pos = i+1
                if pos >= len(general_info):
                    print("Opps... ERROR DETECTED!!!")
                    print("################ bug appears at position 3.... ################")
                    sys.exit(1)
                return general_info[pos].replace('\n', '')
        elif case_part in general_info[i].lower():
            pos = i+1
            if pos >= len(general_info):
                print("Opps... ERROR DETECTED!!!")
                print("################ bug appears at position 4.... ################")
                sys.exit(1)
            return general_info[pos].replace('\n', '')
            


'''
    return catchwords of a case
'''
def get_catchwords(UI = False):
    global catchwords
    if UI:
        print("************************************************")
        print("function calling from UI to get catchwords")
        print("************************************************\n")
    return catchwords


'''
    return case body
'''
def get_body(UI = False):
    global sentences
    if UI:
        print("************************************************")
        print("function calling from UI to get catchwords")
        print("************************************************\n")
    body = ''
    for item in sentences:
        body += item
        body += '\n'
    return body


'''
    find the plaintiff in a case and return
'''    
def get_plaintiff(UI = False):
    # A v B, then A is plaintiff and B is defendent
    if UI:
        print("************************************************")
        print("function calling from UI to get the plaintiff's name")
        print("************************************************\n")
    case_name = get_title()
    plain_and_def = re.split(' [-]*v[-]* ', case_name)
    if len(plain_and_def) == 2:
        return plain_and_def[0]
    else:
        return None


'''
    find the defendent in a case and return 
'''
def get_defendent(UI = False):
    if UI:
        print("************************************************")
        print("function calling from UI to get the defendent's name")
        print("************************************************\n")
    case_name = get_title()
    plain_and_def = re.split(' [-]*v[-]* ', case_name)
    if len(plain_and_def) == 2:
        return plain_and_def[1]
    else:
        return None
    
'''
    return case title
'''
def get_title(UI = False):
    if UI:
        print("************************************************")
        print("function calling from UI to get title of the case")
        print("************************************************\n")
    title = get_general_info('title')
    return title


'''
    return case decision
'''
def get_decision(UI = False):
    if UI:
        print("************************************************")
        print("function calling from UI to get the decision")
        print("************************************************\n")
    decision = get_general_info('decision:')
    return decision


'''
    return the sentences as a list
'''
def get_sentences(UI = False):
    if UI:
        print("************************************************")
        print("function calling from UI to covert the case content into sentences and return a list of sentences")
        print("************************************************\n")
    global sentences
    return sentences
    


'''
    the soup for highlight.py
'''
def get_soup(UI = False):
    global soup
    temp = str(soup)
    temp = BeautifulSoup(temp, "html.parser")
    return temp




'''
    declare some global variables to store the different parts of a case
'''
general_info = []
catchwords = ''
body = ''
plaintiff = ''
defendent = ''
title = ''
decision = ''
sentences = []
soup = ""



