import re
from bs4 import BeautifulSoup


sentence = '''
<div class="" id="article-view-content-div" itemprop="articleBody">
	
		
		
	(서울=연합인포맥스) 신은실 기자 = 삼성선물이 미국 증시 3대 지수의 마이크로 상품에 대해 수수료 무료 이벤트를 진행한다고 5일 밝혔다.<br/><br/>
오는 8월 말까지 거래하는 고객들은 다우존스 30 산업평균지수와 스탠더드앤드푸어스(S&amp;P) 500지수, 나스닥 지수 등 미국의 3대 지수 상품에 대해 무료 수수료(유관기관 비용 제외)가 적용된다.<br/><br/>
이번 이벤트 기간에 해당 상품을 거래하면 연말까지 같은 품목거래에서 기본 수수료를 1달러로 적용받을 수 있다.<br/><br/>
삼성선물의 수수료 무료 이벤트는 이번이 두 번째다.<br/><br/>
삼성선물은 최근 국내 주식시장의 낮은 변동성과 수익률로 개인 투자자들의 해외선물 관심도가 증가하고 있다고 진단했다.<br/><br/>
삼성선물 관계자는 "뉴욕증시에 상장된 구글, 애플, 아마존, 페이스북 등 해외주식투자가 증가하는 만큼 개별종목이 아닌 주요 종목들로 구성된 지수 상품에 대한 관심 역시 증가하고 있다"며 "이벤트를 통해 해외선물을 처음 시작하는 분들에게 수수료 부담을 덜어드리고자 다시 한번 이벤트를 진행하게 됐다"고 말했다.<br/><br/>
esshin@yna.co.kr<br/><br/>
(끝)<br/><br/>본 기사는 인포맥스 금융정보 단말기로 16시 28분에 서비스된 기사입니다.<br/>인포맥스 금융정보 서비스 문의 (398-5209)

		
		<div class="view-copyright">저작권자 © 연합인포맥스 무단전재 및 재배포 금지</div>
<div class="view-editors">
<div class="float-right">
<div class="profile-images">
<img alt="신은실 기자" src="/image/newsroom/default-user.png"/>
</div>
<a class="names" href="mailto:esshin@yna.co.kr"><strong>신은실 기자</strong></a>
</div>
</div>
</div>
'''

soup = BeautifulSoup(sentence, 'html.parser')
pattern1 = re.compile(r'\s+') #공백제거 정규표현식
pattern2 = re.compile('[^ ㄱ-ㅣ가-힣]') #한글 정규표현식
pattern3 = re.compile(r'\t')
pattern4 = re.compile(r'\n')

sentence = re.sub('<.+?>', '', sentence, 0, re.I|re.S)
sentence = re.sub(pattern3, '', sentence)

list1 = sentence.split('\n')
print(list1)