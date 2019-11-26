from selenium import webdriver
driver_path = '../resources/chromedriver'  # driver path 내 위 드라이버로 갔다가 resources밑으로 가서 그 안에 chromedriver가 디렉토리이다.
url = 'https://play.google.com/store/apps/top/category/GAME'
browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
browser.get(url)
#browser.quit()
