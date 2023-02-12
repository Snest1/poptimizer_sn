import undetected_chromedriver
import time
import pandas as pd

from selenium import webdriver
### sudo snap install chromium
### sudo snap refresh chromium
### sudo snap refresh chromium --candidate

def div_load_inet(ticker: str) -> str:
    """Загружает страницу с дивидендами и прочей информацией из интернета"""
    url = 'https://invest' + 'mint.ru/' + ticker + '/'

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
    #/html/body/div[3]/div/div[2]/div[4]

    driver.close()
    driver.quit()
    return content


def div_load_cache(ticker: str) -> list[str] | None:
    try:
        with open("/home/sn/sn/poptimizer-master/auto/cache/" + time.strftime('%Y%m%d',
                                                                             time.gmtime()) + "/" + ticker + '.html',
                 'r') as file:
            lines = [line.rstrip() for line in file]
        file.close()
        return lines
    except:
        print("No cache")
        return None

def div_save_cache(ticker: str, content: any) -> bool:
    try:
        file = open("/home/sn/sn/poptimizer-master/auto/cache/" + time.strftime('%Y%m%d',
                                                                             time.gmtime()) + "/" + ticker + '.html',
                 'w')
        file.write(content)
        file.close()
        return True
    except:
        print("Error saving cache")
        return False


ticker = 'agro'
html = div_load_cache(ticker)
if not html:
#    global html
    html = div_load_inet(ticker)
    div_save_cache(ticker, html)
    html = div_load_cache(ticker)
#print(html)


from bs4 import BeautifulSoup          # https://python-scripts.com/beautifulsoup-parsing
soup = BeautifulSoup('\n'.join(html), "lxml")
#print(soup.prettify())
div_history = soup.find('div', {'id': 'history'})
#print(div_history.prettify())
div_table = div_history.find('table')
#print(str(div_table.prettify()))


from poptimizer.data.adapters.gateways import invest_mint
from poptimizer.data.adapters.html import description

df1 = invest_mint.parser.get_df_from_html(str(div_table), 0, invest_mint.get_col_desc(ticker))
df = description.reformat_df_with_cur(df1, ticker)
print(df)

quit()



html = div_load_inet(ticker)
div_save_cache(ticker, html)
quit()

