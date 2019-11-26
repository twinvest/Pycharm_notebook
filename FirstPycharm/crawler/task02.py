from selenium import webdriver
driver_path = '../resources/chromedriver'  # driver path
urls = [
"https://play.google.com/store/apps/category/GAME_EDUCATIONAL",  # categories
"https://play.google.com/store/apps/category/GAME_WORD",
]
browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
for url in urls:
    browser.get(url)
    #browser.quit()