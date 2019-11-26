from selenium import webdriver
from bs4 import BeautifulSoup
import re
twurl = "https://play.google.com"
def extraction(twurl2):
    browser.get(twurl2)
    page = browser.page_source
    soup = BeautifulSoup(page, "html.parser")
    links = soup.find_all('meta', {'itemprop': 'description'})  # div태그에 class id가W9yFB
    for link in links:
        print(link)
    # hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    # meta_tag = soup.find_all('meta')
    # for select in meta_tag:
    #     select = str(select)
    #     test = hangul.search(select)
    #     if test != None:
    #         print(select)
    # # print(meta_tag)
    #exit()

def search():
    page = browser.page_source
    links = soup.find_all('div', {'class': 'WHE7ib'})  # div태그에 class id가W9yFB
    for link2 in links:
        new_url = link2.a['href']  # a태그중에 href
        new_url = twurl + new_url
        extraction(new_url)
        #browser.get(new_url)


driver_path = '../resources/chromedriver'  # driver path
url = 'https://play.google.com/store/apps/top/category/GAME'
browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
browser.get(url)
page = browser.page_source
soup = BeautifulSoup(page, "html.parser")

links = soup.find_all('div', {'class': 'W9yFB'})  #div태그에 class id가W9yFB
for link in links:
    new_url = link.a['href'] #a태그중에 href
    print(new_url)
    #browser.get(new_url)
    #search()
    #browser.quit()