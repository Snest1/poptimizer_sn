import os
import time
import pandas as pd
import undetected_chromedriver

from selenium import webdriver
# sudo snap install chromium
# sudo snap refresh chromium
# sudo snap refresh chromium --candidate

from bs4 import BeautifulSoup  # https://python-scripts.com/beautifulsoup-parsing
from poptimizer.data.adapters.gateways import invest_mint
from poptimizer.data.adapters.html import description


def div_load_frominet(ticker: str) -> str:
    """Загружает страницу с дивидендами и прочей информацией из интернета"""
    url = 'https://invest' + 'mint.ru/' + ticker + '/'

    service_log_path = "chromedriver.log"
    service_args = ['--verbose']

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
#    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver = undetected_chromedriver.Chrome(executable_path='chromedriver',
                                            service_args=service_args,
                                            service_log_path=service_log_path,
                                            options=chrome_options)
    driver.get(url)
    content = driver.page_source

# driver.save_screenshot('/home/sn/sn/poptimizer-master/auto/scr_'+time.strftime('%Y%m%d', time.gmtime()) +'.png')

    driver.close()
    driver.quit()
    return content


def div_load_fromcache(ticker: str) -> list[str] | None:
    try:
        with open("/home/sn/sn/poptimizer-master/auto/cache/" +
                  time.strftime('%Y%m%d', time.gmtime()) + "/" +
                  ticker + '.html', 'r') as file:
            lines = [line.rstrip() for line in file]
        file.close()
        return lines
    except Exception as e:
        print(f"No cache: {e}")
        return None


def div_save_cache(ticker: str, content: any) -> bool:
    try:
        filename = "/home/sn/sn/poptimizer-master/auto/cache/" + \
                   time.strftime('%Y%m%d', time.gmtime()) + "/" + ticker + '.html'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file = open(filename, 'w')
        file.write(content)
        file.close()
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False


def div_get(ticker: str, use_cache: bool) -> pd.DataFrame:
    html = None
    if use_cache:
        html = div_load_fromcache(ticker)
    if not html:
        html = div_load_frominet(ticker)
        div_save_cache(ticker, html)
        html = div_load_fromcache(ticker)  # Это нужно, чтобы далее html был list of string

    soup = BeautifulSoup('\n'.join(html), "lxml")
#    print(soup.prettify())
    div_history = soup.find('div', {'id': 'history'})
#    print(div_history.prettify())
    div_table = div_history.find('table')
#    print(str(div_table.prettify()))

    df1 = invest_mint.parser.get_df_from_html(str(div_table), 0, invest_mint.get_col_desc(ticker))
    df = description.reformat_df_with_cur(df1, ticker)
    return df


div_df = div_get('magn', use_cache=True)
print(div_df)
quit()
