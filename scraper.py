# This is for functional test.

from predict_BayesianModel import retTopK, labelTopK
from analyse_url_new import *
from train_BayesianModel import demo_training_bayesianModel


def getHello():
  return("HELLO WORLD")




url = "http://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWCATCD/2017/2.html"
get_details(url)
case_title = get_title()
plaintiff = get_plaintiff()
defendent = get_defendent()
case_content = get_body()
case_decision = get_decision()
case_catchword = get_catchwords()
all_sentences = get_sentences()

topK_sentences_index = retTopK(url)
topK_sentences = labelTopK(topK_sentences_index, url, 5)
#for i in retTopK(url):
#    topK_sentences.append(all_sentences[i])


print("************************************************")
print("case_title *************************************")
print("************************************************")
print(case_title)
print("************************************************")
print("plaintiff **************************************")
print("************************************************")
print(plaintiff)
print("************************************************")
print("defendent **************************************")
print("************************************************")
print(defendent)
print("************************************************")
print("case_decision **********************************")
print("************************************************")
print(case_decision)
print("************************************************")
print("case_catchword *********************************")
print("************************************************")
print(case_catchword)
print("************************************************")
print("************************************************")
print("case_content ***********************************")
print(case_content)
print("************************************************")
print("************************************************")
print("topK_sentences *********************************")
print("************************************************")
print(topK_sentences)

print("************************************************")
print("demo the training steps ************************")
print("************************************************")
demo_training_bayesianModel()

