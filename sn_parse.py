from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options 

from time import sleep

import undetected_chromedriver

url = 'https://invest' + 'mint.ru/mdmg/'


service_log_path = "chromedriver.log"
service_args = ['--verbose']


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("no-sandbox")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
#chrome_options.add_argument("--disable-blink-features=AutomationControlled")

driver = undetected_chromedriver.Chrome(executable_path='chromedriver',
        service_args=service_args,
        service_log_path=service_log_path,
        options=chrome_options)
driver.get(url)
content = driver.page_source
print(content)

#/html/body/div[3]/div/div[2]/div[4]


driver.close()
driver.quit()
quit()





from bs4 import BeautifulSoup
import requests
#url = 'http://mignews.com/mobile'
url = 'https://investmint.ru/mdmg/'
page = requests.get(url)

#Проверим подключение:
print(page.status_code)

new_news = []
news = []

#Самое время воспользоваться BeautifulSoup4 и скормить ему наш page, 
#указав в кавычках как он нам поможет 'html.parcer':
soup = BeautifulSoup(page.text, "html.parser")

#Если попросить его показать, что он там сохранил:
print(soup)

quit()

#Теперь воспользуемся функцией поиска в BeautifulSoup4:
news = soup.findAll('a', class_='lenta')

for news_item in news:
    if news_item.find('span', class_='time2 time3') is not None:
        new_news.append(news_item.text)

print(f"{news_item =}")