import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from matplotlib import pylab as plt
import numpy as np
import pickle as pk
import jieba
from functools import reduce
import operator as op
from tqdm import tqdm
from easydict import EasyDict as edict
import newspaper
from newspaper import Article
from newspaper import Config
cupoy_url = "https://www.cupoy.com/newsfeed/topstory"
max_news  = 500
ckpt_web        = False
web_extract     = False
article_extract = False
if web_extract:
    browser = webdriver.Chrome(executable_path='/Users/vincentwu/Documents/GitHub/2nd-PyCrawlerMarathon/homework/driver/chromedriver')
    browser.get(cupoy_url)
    time.sleep(15)

news_dict = edict()
news_dict.title =   []
news_dict.catagory =   []
news_dict.address = []
print_len = 0
#################################################################################################
# TODO: Web extract                                                                             #
#################################################################################################
while web_extract:
    html = browser.page_source
    soup = BeautifulSoup(html, 'lxml')
    news = soup.find("div", class_="ReactVirtualized__Grid__innerScrollContainer").find_all("div", class_="sc-eEieub sc-iuDHTM emMJDZ")
    for ni in news:
        #title   = ni.find("h6", class_="sc-erNlkL sc-ekulBa gyyrZF").text
        cls     = ni.find("div", class_="sc-gacfCG bPSpUf").text
        addr    = ni.find("a")["href"]
        title   = ni.find("a")["title"]
        if title not in news_dict.title:
            news_dict.title.append(title)
            news_dict.catagory.append(cls)
            news_dict.address.append(addr)
    if len(news_dict.title) >= max_news:
        print ("News count(final): %d" %(len(news_dict.title)))
        break
    if (len(news_dict.title)!=print_len):
        print ("News count: %d" %(len(news_dict.title)))
        print_len = len(news_dict.title)
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")

if ckpt_web and web_extract:
    pk.dump(news_dict, open("proj_data_web.pkl", "wb"))

if (len(news_dict.title) <= 0):
    print ("Loading web data from proj_data_web.pkl!!!")
    news_dict = pk.load(open("proj_data_web.pkl", "rb"))

if len(news_dict.title) > max_news:
    print ("News count before slice: %d" %(len(news_dict.title)))
    news_dict.title     = news_dict.title[:max_news]
    news_dict.catagory  = news_dict.catagory[:max_news]
    news_dict.address   = news_dict.address[:max_news]
    print ("News count after slice: %d" %(len(news_dict.title)))

#################################################################################################
# TODO: Article by newspaper                                                                    #
# ref: https://newspaper.readthedocs.io/en/latest/                                              #
#################################################################################################
if article_extract:
    print ("\n[Extract Article]")
    article_all_list = []
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    failed_count = 0
    for ai in tqdm(news_dict.address):
        try:
            art     = Article(ai, language="zh")
            art.download()
            art.parse()
            article_all_list.append(art.text.replace("\n", ""))
        except:
            failed_count += 1
            print ("---[Failed extract]: ", ai)
    failed_rate = float(failed_count) / float(max_news)
    print ("[Extract Article report]")
    print ("\t sucess rate: %d %%\n" %((1-failed_rate)*100))
    print ("\t failed rate: %d %%\n" %(failed_rate*100))
        #########################################################
        # BeautifulSoup caused None when art_list += art        #
        #########################################################
        #art_list= ""
        #res     = requests.get(ai)
        #soup    = BeautifulSoup(res.text, "html.parser")
        #time.sleep(0.1)
        #article = soup.find_all("p")
        #for art in article:
        #    art_list += art.test
        #article_all_list.append(art_list)
    news_dict.article = article_all_list
if article_extract:
    pk.dump(news_dict, open("proj_data_art.pkl", "wb"))
else:
    print ("Loading article data from proj_data_art.pkl!!!")
    news_dict = pk.load(open("proj_data_art.pkl", "rb"))

#################################################################################################
# TODO: Pie table plot by matplotlib                                                            #
# ref: https://blog.csdn.net/gmr2453929471/article/details/78655834                             #
# ref: https://codertw.com/程式語言/359974/                                                     #
# ref: https://www.itread01.com/content/1545238867.html                                         #
# ref: https://ithelp.ithome.com.tw/articles/10196410                                           #
# ref: https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py                                                                  #
#################################################################################################
news_pd = pd.DataFrame({"title": news_dict.title, "catagory": news_dict.catagory})
news_gp = news_pd.groupby("catagory")
news_count_dict = news_gp.count().to_dict()["title"]
max_pie_out     = 10
max_explode     = 3
news_cls_list   = list(news_count_dict.keys())  [:max_pie_out]
news_count_list = list(news_count_dict.values())[:max_pie_out]
ii              = np.argsort(news_count_list)[::-1]
news_cls_arr    = np.array(news_cls_list)[ii]
news_count_arr  = np.array(news_count_list)[ii]
pie_explode     = np.zeros_like(ii)
pie_explode[:max_explode]  = 0.0
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/Users/vincentwu/Downloads/simheittf/simhei.ttf',size=10)
#ax1.set_xticklabels(ability_labels,fontproperties=font)
patches, l_text, p_text = plt.pie(news_count_arr, labels=news_cls_arr, autopct="%1.1f%%",explode=pie_explode, radius=1)
for li in l_text:
    li.set_fontproperties(font)
plt.axis("equal")
plt.savefig("proj_wordcloud.png")
print ("[Pie chart]:\n\tsave png to proj_pie.png")

#################################################################################################
# TODO: jieba                                                                                   #
# ref: https://github.com/tomlinNTUB/Python-in-5-days/blob/master/10-2%20中文斷詞-移除停用詞.md #
#################################################################################################
stop_words = []
file_stop_words = "jieba_stop_words.txt"
with open(file_stop_words, 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stop_words.append(data)
##title_all = [ti+"。" for ti in news_dict.title]
##title_all = reduce(op.add, title_all)
words_all  = []
words_all2 = []
print ("[Cut sentence by Jieba]")
for ti in tqdm(news_dict.article):
    if len(ti) <= 0: continue
    segments = jieba.cut(ti, cut_all=False)
    remainderWords = list(filter(lambda a: a not in stop_words and a != '\n', segments))
    words_all2 += remainderWords
    remainderWords = [wi.strip() + " " for wi in remainderWords]
    words_all += [reduce(op.add, remainderWords)]
words_pd  = pd.DataFrame(words_all2, columns=["word"])
words_gp  = pd.DataFrame(words_pd.groupby("word").size(), columns=["cumsum"])
words_gp  = words_gp.sort_values(by="cumsum", ascending=False)
#################################################################################################
# TODO: TFIDF                                                                                   #
# ref:  https://blog.csdn.net/Eastmount/article/details/50323063                                #
# ref:  https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfT#
#       ransformer.html#sklearn.feature_extraction.text.TfidfTransformer                        #
#################################################################################################
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
pipe = Pipeline([('count', CountVectorizer(vocabulary=stop_words)),('tfid', TfidfTransformer())]).fit(news_dict.article)
ff   = open("proj_tfidf.txt", "w")
ii   = np.argsort(pipe['tfid'].idf_)
for iii in ii:
    ff.write("%s: \t%.4f\n" %(stop_words[iii], pipe['tfid'].idf_[iii]))
ff.close()
import pdb; pdb.set_trace()
print ("[TFIDF]:\n\tsave to file (proj_tfidf.txt) !")
#vectorizer = CountVectorizer()
#transformer = TfidfTransformer()
#
#word = vectorizer.get_feature_names()
#weight=tfidf.toarray()
#for i in range(len(weight)):
#    print ("-------The TFIDF weight for class %d -----" %i)
#    for j in range(len(word)):
#        if float(weight[i][j]) > 0:
#            print (word[j],weight[i][j])

#################################################################################################
# TODO: words cloud                                                                             #
# ref:  http://stacepsho.blogspot.com/2018/06/word-cloud-in-python.html                         #
#################################################################################################
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter
from matplotlib.font_manager import fontManager

my_wordcloud = WordCloud(background_color="white",collocations=False, width=2400, height=2400, margin=2, font_path='/Users/vincentwu/Downloads/simheittf/simhei.ttf')
my_wordcloud.generate_from_frequencies(frequencies=Counter(words_all2))

#產生圖片
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(my_wordcloud,interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
#顯示用
plt.savefig("proj_wordcloud.png")
print ("[wordcloud]:\n\tsave png to proj_wordcloud.png")
#plt.show()
