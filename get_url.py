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

Module to get all URL's of project scope: 
NSW Civil and Administrative Tribuneral

How to run:
get_case_doc_urls() gets list of case law document urls in a text file.
get_catchwords(url) takes in url and outputs catchwords to console. Also returns true/false.

Created on Tue Aug 22 19:19:42 2017

@author: Jeff
"""

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re

global_list = []

# 6 minutes for 50 links.
def get_links_demo_scope(output_path):
    url_original = 'http://www.austlii.edu.au'
    #url_list = []
    url_scope = 'http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/'
    
    with open(output_path, "w+") as text_file:
    
        req = Request(url_scope, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.90 Safari/537.36'})
        html = urlopen(req).read()
        x=0
        for link in BeautifulSoup(html, 'lxml', parse_only=SoupStrainer('a')):
            if hasattr(link, 'href'):
                if '/cgi-bin/viewdoc/' in link['href']:
                    url = url_original+link['href']
                    #url_list.append(url)
                    #global_list.append(url)
                    if get_catchwords(url):
                        #url = url+'\n'
                        #f.write(url)
                        print("{}".format(url), file=text_file)
                        print('Good link {} retrieved...'.format(x))
                        # Get 5 good links from the 50.
                        if x == 5:
                            break
                        x = x + 1
                        '''
                        print('hello world.')
                    else:
                        print('bad link.')
                        '''
                    #f.write(url)
    #f.close() 
    #return url_list

def get_links_demo_UI(output_path):
    url_original = 'http://www.austlii.edu.au'
    #url_list = []
    url_scope = 'http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/'
    
    with open(output_path, "w+") as text_file:
    
        req = Request(url_scope, headers={'USer-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        for link in BeautifulSoup(html, 'lxml', parse_only=SoupStrainer('a')):
            if hasattr(link, 'href'):
                if '/cgi-bin/viewdoc/' in link['href']:
                    url = url_original+link['href']
                    #url_list.append(url)
                    #global_list.append(url)
                    if get_catchwords(url):
                        #url = url+'\n'
                        #f.write(url)
                        return "{}".format(url)
                        break

# arg1 is a list of url strings #DONT NEED 
def check_catchwords_exist():
    get_links_shorter_scope('./output/Case_Links.txt')
    for case in global_list:
        #search_catchwords(case)
        
        if get_catchwords(case):
            print('hello world.')
        else:
            print('bad link.')
        
'''
Function to check each link for catchwords.
Only cases with catchwords exceeding 50 characters are good links. 
NO LONGER NEED
'''
def get_good_case_urls():
    
    check_catchwords_exist()
# using requests and re instead of beautifulsoup #DONT NEED, no big difference
def search_catchwords(arg1):
    page = requests.get(arg1)
    regex = '<div>Catchwords: </div>'
    one = re.findall(regex, page.text)[0]
    print(one)

##### Original, part of get_case_scope_urls() and get_case_doc_urls()
def get_links(url, path_match):
    url_original = 'http://www.austlii.edu.au'
    url_scope_list = []
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    conn = urlopen(req)
    response = conn.read()
        
    for link in BeautifulSoup(response, 'lxml', parse_only=SoupStrainer('a')):
        if hasattr(link, 'href'):
            #print(link['href'])
            if path_match in link['href']:
                if len(path_match) == 3:        # i.e. matching "201"
                    if len(link['href']) == 5:  # problematic 2017/ etc
                        url3 = url+link['href']
                        url_scope_list.append(url3)
                        #print('This: {}'.format(url3))
                elif '.html' in path_match:     # special case for getting docs.
                    if '201' in link['href']:   # gets around matching problem
                        url3 = url_original+link['href']
                        url_scope_list.append(url3)
                        #print('That: {}'.format(link['href']))
                else:
                    url3 = url_original+link['href']
                    url_scope_list.append(url3)
                    #print('That: {}'.format(link['href']))
                    
    return url_scope_list

# Original, used in get_case_doc_urls()
def get_case_scope_urls():
    url_scope_list = []
    url = 'http://www.austlii.edu.au'
    url2 = url+'/databases.html'        # contains the links to different case types
    url_scope_list = get_links(url2, 'au/cases/nsw/NSWCATCD')
    return url_scope_list

# Original, main calling function.
def get_case_doc_urls(output_path, num):
    url_scope_list = []
    url_scope_list = get_case_scope_urls()
    num = int(num)
    x=1
    #originally saved as Case_Links2.txt
    with open(output_path, "w") as text_file:      # open the text file here
 
        for case_scope_url in url_scope_list:
            #print(case_scope_url)
            case_year_url_list = get_links(case_scope_url, '201')
            for case_year_url in case_year_url_list:
                #print(case_year_url)
                url_list = get_links(case_year_url, '.html')
                for case in url_list:
                    #if check_catchwords(case):
                    if get_catchwords(case):
                        #url = url+'\n'
                        #f.write(url)
                        print("{}".format(case), file=text_file)
                        print('Good link {} retrieved...'.format(x))
                        # Get 5 good links from the 50.
                        if x >= num:
                            return
                        x = x + 1
                    #print(case)
                    #print("{}".format(case), file=text_file)
        
# obsolete
def check_catchwords(arg1):
    # open arg1 case document page, check for catchphrase
    req = Request(arg1, headers={'User-Agent': 'Mozilla/5.0'})
    conn = urlopen(req)
    response = conn.read()
    #trs = document.getElementsByTagName('tr')
    soup = BeautifulSoup(response, 'lxml')
    td_tag = soup.find_all('td', {'width':'205'})
    for td in td_tag:
        if td.find('div'):
            if "Catchwords:" in td.find('div').text:
                return True
    return False
        #if 'Catchwords' in link:
            #print(tr)

# Returns string if catchwords present. else False. Jeff may want list.
# Able to return catchwords with slight modification
def get_catchwords(arg1):
    req = Request(arg1, headers={'User-Agent': 'Mozilla/5.0'})
    conn = urlopen(req)
    html = conn.read()
    soup = BeautifulSoup(html, 'lxml')
    all_td = soup.find_all('td')#,{'width':'205'})
    # Just gets catchwords immediately after the catchwords title.
    x=0
    for td in all_td:
        if x == 1:
            x = 0
            if(len(td.find('div').text) > 50):
                
            #print(td.find('div').text)
                return True
            #return td.find('div').text
        if td.find('div'):
            if "Catchwords:" in td.find('div').text:
                x=1;
    return False
        
# Able to get case details.
def get_case_details(arg1):
    req = Request(arg1, headers={'User-Agent': 'Mozilla/5.0'})
    conn = urlopen(req)
    html = conn.read()
    soup = BeautifulSoup(html, 'lxml')
    all_li = soup.find_all('li',{'style':'text-indent:0pt; margin-top:0pt; margin-bottom:0pt;'})
    # Just gets catchwords immediately after the catchwords title.
    for li in all_li:
        print(li)
        #print(li.text)
    
    
