from bs4 import BeautifulSoup
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
soup = BeautifulSoup(html_doc, 'html.parser') #beautiful soup으로 받아서 html.paser로 정리한다. 그리고 soup에 저장.
#print(soup.prettify()) #prettify는 예쁘게 정리해주는 함수


#예제 3-1코드
tag = soup.a #a태그를 의미
# print(tag)
# print(tag.name)
# print(tag.attrs) #{'href': 'http://example.com/elsie', 'class': ['sister'], 'id': 'link1'}
# print(tag.string) #Elsie
# print(tag['class']) #['sister']
# print(soup.title) #<title>The Dormouse's story</title>
# print(soup.title.name) #title
# print(soup.title.string) #The Dormouse's story
# print(soup.title.parent.name)  # parent
# print(soup.title.parent.title.string)
# print(soup.head.contents[0].string)  # contents : children as a list. 즉, title의 스트링임. 위와 같은 결과.
# print(soup.p) # 이렇게하면 하나만 가지고온다.
# print(soup.p['class']) #class의 value는 title.
# print(soup.a) #이렇게하면 제일 위 a태그 하나만 가지고온다.
# print(soup.find_all('a')) #a태그인 모든것을 가지고온다.
# print(soup.find(id='link3')) #세번째 a태그
# print(soup.find(id='link3').string) #세번째 a태그의 내용인 Tillie
# print(soup.find_all('a'))
# for link in soup.find_all('a'):
#      print(link)
#      print(link.attrs)
#
#      print(link.get('href'))
#      print(link['href'])
print(soup.get_text()) #태그와 태그사이에 있는 텍스트를 전부 가지고온다.