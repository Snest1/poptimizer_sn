"""Загрузка различных данных с MOEX."""
from typing import Dict, List, Optional, Union

import aiomoex
import pandas as pd

from poptimizer.data.adapters.gateways import gateways
from poptimizer.shared import adapters, col


class TradingDatesGateway(gateways.BaseGateway):
    """Обновление для таблиц с диапазоном доступных торговых дат."""

    _logger = adapters.AsyncLogger()

    async def __call__(self) -> pd.DataFrame:
        """Получение обновленных данных о доступном диапазоне торговых дат."""
        self._logger("Загрузка данных по торговым дням")
        json = await aiomoex.get_board_dates(
            self._session,
            board="TQBR",
            market="shares",
            engine="stock",
        )
        return pd.DataFrame(json, dtype="datetime64[ns]")


class IndexesGateway(gateways.BaseGateway):
    """Котировки различных индексов на MOEX."""

    _logger = adapters.AsyncLogger()

    async def __call__(
        self,
        ticker: str,
        start_date: Optional[str],
        last_date: str,
    ) -> pd.DataFrame:
        """Получение значений индекса на закрытие для диапазона дат."""
        self._logger(f"{ticker}({start_date}, {last_date})")
        json = await aiomoex.get_market_history(
            session=self._session,
            start=start_date,
            end=last_date,
            security=ticker,
            columns=("TRADEDATE", "CLOSE"),
            market="index",
        )
        df = pd.DataFrame(json)
        df.columns = [col.DATE, col.CLOSE]
        df[col.DATE] = pd.to_datetime(df[col.DATE])
        return df.set_index(col.DATE)


class SecuritiesGateway(gateways.BaseGateway):
    """Информация о всех торгующихся акциях."""

    _logger = adapters.AsyncLogger()

    async def __call__(self, market: str, board: str) -> pd.DataFrame:
        """Получение списка торгуемых акций с ISIN и размером лота."""
#SNEDIT        self._logger("Загрузка данных по торгуемым бумагам")
        self._logger(f"SNLOG_01. Загрузка данных по торгуемым бумагам. Market={market}. Board={board}")

        columns = (
            "SECID",
            "ISIN",
            "LOTSIZE",
            "SECTYPE",
        )
        json = await aiomoex.get_board_securities(
            self._session,
            market=market,
            board=board,
            columns=columns,
        )
        df = pd.DataFrame(json)
        df.columns = [col.TICKER, col.ISIN, col.LOT_SIZE, col.TICKER_TYPE]

#        self._logger(f"SNLOG_01. df={df}")

#SNADDED
        if (df.query('TICKER == "GLDRUB_TOM"').empty == True and df.query('TICKER == "GAZP"').empty == False):
#            self._logger(f"SNLOG_01. Try: add GLDRUB_TOM to securitye")
            df = df.append({ 'TICKER':'GLDRUB_TOM', 'ISIN':'RU_GLDRUB_TOM', 'LOT_SIZE' : 1,     'TICKER_TYPE' : '1'}, ignore_index=True)
            df = df.append({ 'TICKER':'SLVRUB_TOM', 'ISIN':'RU_SLVRUB_TOM', 'LOT_SIZE' : 1,     'TICKER_TYPE' : '1'}, ignore_index=True)
            df = df.append({ 'TICKER':'BTCRUB',     'ISIN':'RU_BTCRUB',     'LOT_SIZE' : 1,     'TICKER_TYPE' : '1'}, ignore_index=True)
            df = df.append({ 'TICKER':'ETHRUB',     'ISIN':'RU_ETHRUB',     'LOT_SIZE' : 1,     'TICKER_TYPE' : '1'}, ignore_index=True)

#        self._logger(f"SNLOG_01. after adding")


        return df.set_index(col.TICKER)


class AliasesGateway(gateways.BaseGateway):
    """Ищет все тикеры с эквивалентным регистрационным номером."""

    _logger = adapters.AsyncLogger()

    async def __call__(self, isin: str) -> List[str]:
        """Ищет все тикеры с эквивалентным ISIN."""
        self._logger(isin)

        json = await aiomoex.find_securities(self._session, isin, columns=("secid", "isin"))
        return [row["secid"] for row in json if row["isin"] == isin]


IISJson = List[Dict[str, Union[str, int, float]]]


def _format_candles_df(json: IISJson) -> pd.DataFrame:
    """Так как может прийти пустой json, необходимо принудительно создать наименование колонок."""
    df = pd.DataFrame(json)
    columns = [
        "begin",
        "open",
        "close",
        "high",
        "low",
        "value",
        "end",
        "volume",
    ]
    df = df.reindex(columns, axis=1)
    df = df.drop(["end", "volume"], axis=1)
    df.columns = [
        col.DATE,
        col.OPEN,
        col.CLOSE,
        col.HIGH,
        col.LOW,
        col.TURNOVER,
    ]
    df[col.DATE] = pd.to_datetime(df[col.DATE])

    return df.set_index(col.DATE)


class QuotesGateway(gateways.BaseGateway):
    """Загружает котировки акций."""

    _logger = adapters.AsyncLogger()

    async def __call__(
        self,
        ticker: str,
        market: str,
        start_date: Optional[str],
        last_date: str,
    ) -> pd.DataFrame:
        """Получение котировок акций в формате OCHLV."""
        self._logger(f"{ticker}({start_date}, {last_date}) --START")

        json = await aiomoex.get_market_candles(
            self._session,
            ticker,
            market=market,
            start=start_date,
            end=last_date,
        )
        self._logger(f"{ticker}({start_date}, {last_date}) --END")


#SNEDIT_ADD

        if (ticker == 'YNDX'): 

### Создадим признак, что курсы были загружены  2023

            from pymongo.collection import Collection
            from poptimizer.store.database import DB, MONGO_CLIENT
            misc_collection = MONGO_CLIENT[DB]['misc']

            import datetime
            post = {"_id": "need_update_TV",
              "data": datetime.datetime.utcnow(),
              "timestamp": datetime.datetime.utcnow()}

            misc_collection.replace_one(filter={"_id": "need_update_TV"}, replacement=post, upsert=True)




#SNEDIT_ADD

# Надо из моих тикеров удалить данные за даты бОльшие, чем есть информация с моекс. например по газпрому.
#        if (ticker == 'GAZP'):
        if (ticker == 'YNDX'):   # Надо позднее начинать, чтобы дата начальных тикетов успела сохраниться

### Создадим признак, что курсы были загружены  2023

            from pymongo.collection import Collection
            from poptimizer.store.database import DB, MONGO_CLIENT
            misc_collection = MONGO_CLIENT[DB]['misc']

            import datetime
            post = {"_id": "need_update_TV",
              "data": datetime.datetime.utcnow(),
              "timestamp": datetime.datetime.utcnow()}

            misc_collection.replace_one(filter={"_id": "need_update_TV"}, replacement=post, upsert=True)

#    collection_speedy.delete_one({'_id': speedy_id.inserted_id})

##################################################



##            gazp = _format_candles_df(json)
#
##  загрузим список тикеров и дат, т.к. другие библиотеки poptimizer не работают
#            from pymongo import MongoClient
#            quotes_collection = MongoClient('localhost', 27017)['data']['quotes']
#            sn_tickers = []   # список
#            sn_dates = []   # список
#            for quote in quotes_collection.find():
#                sn_tickers.append(quote['_id'])
#                for one_date in quote['data']['index']:
#                    if one_date not in sn_dates:
#                        sn_dates.append(one_date)
#            sn_dates.sort()
#            print(sn_dates)
#
#
#            LoadFromTV('GLDRUB_TOM', "MOEX:GLDRUB_TOM", 1, sn_dates)
#            LoadFromTV('SLVRUB_TOM', "MOEX:SLVRUB_TOM", 1, sn_dates)
#            LoadFromTV('BTCRUB', "BINANCE:BTCRUB/10000", 10000, sn_dates)
#            LoadFromTV('ETHRUB', "BINANCE:ETHRUB/100", 100, sn_dates)
##            quit()
#
#
#
##        if (ticker == 'FXGD'):
##
##            import datetime
##            bars = datetime.datetime.strptime(last_date, "%Y-%m-%d").date() - datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
##
##
##            self._logger(f"SNLOG_02. FXGD Here. start={start_date}; end={last_date}; bars={bars.days}")
##            self._logger(f"SNLOG_02. JSON={json}")
##            temp_df = _format_candles_df(json)
##            self._logger(f"SNLOG_02. DF={temp_df}")
##
##            from tvDatafeed import TvDatafeed,Interval
##            tv = TvDatafeed()
##            tv.clear_cache()
##            t_FXGD = tv.get_hist("MOEX:FXGD", interval=Interval.in_daily,n_bars=bars.days)
##            self._logger(f"SNLOG_03. TV={t_FXGD}")


        return _format_candles_df(json)


class USDGateway(gateways.BaseGateway):
    """Курс доллара."""

    _logger = adapters.AsyncLogger()

    async def __call__(
        self,
        start_date: Optional[str],
        last_date: str,
    ) -> pd.DataFrame:
        """Получение значений курса для диапазона дат."""
        self._logger(f"({start_date}, {last_date})")
        self._logger(f"SNLOG_03 USD here. ")   #SNEDIT_ADD

        json = await aiomoex.get_market_candles(
            self._session,
            "USD000UTSTOM",
            market="selt",
            engine="currency",
            start=start_date,
            end=last_date,
        )

        return _format_candles_df(json)
