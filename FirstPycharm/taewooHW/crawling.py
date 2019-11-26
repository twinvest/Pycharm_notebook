from selenium import webdriver
from bs4 import BeautifulSoup
import re
import operator
from collections import defaultdict
from konlpy.tag import Hannanum
from collections import Counter
import pytagcloud # Add fonts supporting Korean

words_dict = defaultdict(int)#전역변수로두자.


def wordcounting(lines):
    list = lines.split('\n')

    for sentence in list:  # 리스트 형태의 문장을 for문을 통해 하나씩 가져와서
        if(sentence == '' or sentence=='\n' or sentence=='\t' or sentence ==' '):
            continue #리스트에 만약 공백이 들어있다면 continue
        sentence.strip()
        for word in sentence.split(' '):  # 띄어쓰기 단위로 나누고
            if (word == '' or word == '\n' or word == '\t' or word == ' '):
                continue  # 리스트에 만약 공백이 들어있다면 continue
            words_dict[word] += 1  # 해당 키의 벨류값을 1씩 증가시킨다.
    #print(words_dict)
    #print(sorted(words_dict))

    # sdict = sorted(words_dict.items(), key=operator.itemgetter(1))
    # for i in sdict:
    #      if i[1] >= 10:
    #          print(i)
    print('=============================')

def extraction(sentence):
    soup = BeautifulSoup(sentence, 'html.parser')
    pattern1 = re.compile(r'\s+')  # 공백제거 정규표현식
    pattern2 = re.compile('[^ ㄱ-ㅣ가-힣]')  # 한글 정규표현식
    pattern3 = re.compile(r'\t')
    sentence = re.sub('<.+?>', '', sentence, 0, re.I | re.S)
    # print(sentence)
    sentence = re.sub(pattern3, '', sentence)
    wordcounting(sentence)
    #print(sentence)

def search(twurl2):
    browser.get(twurl2)
    page = browser.page_source
    soup = BeautifulSoup(page, "html.parser")
    links = soup.find_all('div', {'id': 'article-view-content-div'})
    for link in links:
        # print(link)
        extraction(str(link))

if __name__ == "__main__":
    url2 = 'http://news.einfomax.co.kr' #연합인포맥스 주소
    driver_path = '../resources/chromedriver'  # driver path
    search_string = input()

    for i in range(0, 1):
        print('=====================================')
        print('page' + str(i+1))
        print('=====================================')
        url = 'http://news.einfomax.co.kr/news/articleList.html?page=' + str(
            i+1) + '&total=164&box_idxno=&sc_area=A&view_type=sm&sc_word=' + search_string
        browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
        browser.get(url)
        page = browser.page_source
        soup = BeautifulSoup(page, "html.parser")

        links = soup.find_all('div', {'class': 'list-block'}) #div태그 중 class값이 list-block인것들
        #print(links)
        for link in links:
            new_url = url2 + link.a['href'] #a태그중에 href
            browser.get(new_url)
            #print(new_url)
            search(new_url)
    print(words_dict)