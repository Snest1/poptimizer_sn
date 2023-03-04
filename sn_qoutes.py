


######### Фильтровать организмы, по которым идет расчет.  Нужно, чтобы статистика популяции вверху отражалась корректно







"""Эволюция параметров модели."""
import datetime
import itertools
import logging
import operator
from typing import Optional, Final

import numpy as np
from scipy import stats

from poptimizer import config
from poptimizer.data.views import listing
from poptimizer.dl import ModelError
from poptimizer.evolve import population, seq
from poptimizer.portfolio.portfolio import load_tickers


population.print_stat()

from poptimizer.evolve import store
import logging
LOGGER = logging.getLogger()


from pymongo.collection import Collection
from poptimizer.store.database import DB, MONGO_CLIENT
_COLLECTION = MONGO_CLIENT[DB]["sn_speedy"]
def snspeedy_get_collection() -> Collection:
    return _COLLECTION


def sn_get_keys(key_1: str, key_2: str = None, key_3: str = None) ->list:
    local_list = []
    cursor = db_find(filter={key_1: {"$exists": True}}, projection=[key_1])
    for document in cursor:
        id = str(document['_id'])
#        print("key_2=",key_2)
#        print("type key_1=",type(document[key_1]))
        if (isinstance(document[key_1], (tuple, list))) and (key_2 is None):
            keys = tuple(document[key_1])
        else:
            if key_3 is not None:
                keys = document[key_1][key_2][key_3]
            elif key_2 is not None:
                keys = document[key_1][key_2]
            else:
                keys = document[key_1]
        local_list.append([id, keys])
#       print(id, "\t", keys[1])
    return local_list


def _select_worst_bound(candidate: dict, metric: str) -> tuple[float, float, float]:
    """Выбирает минимальное значение верхней границы доверительного интервала.

    Если данный организм не уступает целевому организму, то верхняя граница будет положительной.
    """

    diff = _aligned_diff(candidate, metric)

    bounds = map(
        lambda size: _test_diff(diff[:size]),
        range(1, len(diff) + 1),
    )

    return min(
        bounds,
        key=lambda bound: bound[1] or np.inf,
    )


def _aligned_diff(candidate: dict, metric: str) -> list[float]:
    comp = []

    for base in population.get_metrics():
        metrics = base[metric]

        if base["date"] < candidate["date"]:
            metrics = [np.nan] + metrics

        scores = len(candidate[metric])

        metrics = metrics[:scores]
        metrics = metrics + [np.nan] * (scores - len(metrics))

        comp.append(metrics)

    comp = np.nanmedian(np.array(comp), axis=0)

    return list(map(operator.sub, candidate[metric], comp))


def _test_diff(diff: list[float]) -> tuple[float, float, float]:
    """Последовательный тест на медианную разницу с учетом множественного тестирования.

    Тестирование одностороннее, поэтому p-value нужно умножить на 2, но проводится 2 раза.
    """
    _, upper = seq.median_conf_bound(diff, config.P_VALUE / population.count())

    return float(np.median(diff)), upper, np.max(diff)

def sn_scale() -> float:
    return population.count() ** 0.5


def sn_time_delta(org):
    """Штраф за время, если организм медленнее медианного в популяции."""
    times = [doc["timer"] for doc in population.get_metrics()]

    return stats.percentileofscore(times, org.timer, kind="mean") / 100


def sn_get_margin(org: population.Organism) -> tuple[float, float]:
    """Используется тестирование разницы llh и ret против самого старого организма.

    Используются тесты для связанных выборок, поэтому предварительно происходит выравнивание по
    датам и отбрасывание значений не имеющих пары (возможно первое значение и хвост из старых
    значений более старого организма).
    """
    margin = np.inf

    names = {"llh": "LLH", "ir": "RET"}

    for metric in ("llh", "ir"):
        median, upper, maximum = _select_worst_bound(
            candidate={"date": org.date, "llh": org.llh, "ir": org.ir},
            metric=metric,
        )

        print(
            " ".join(
                [
                    f"{names[metric]} worst difference:",
                    f"median - {median:0.4f},",
                    f"upper - {upper:0.4f},",
                    f"max - {maximum:0.4f}",
                ],
            ),
        )

        valid = upper != median
        margin = min(margin, valid and (upper / (upper - median)))

    if margin == np.inf:
        margin = 0

    time_score = sn_time_delta(org)

    print(f"Margin - {margin:.2%}, Slowness - {time_score:.2%}\n")  # noqa: WPS221

    if margin < 0:
        org.die()
        print("НЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕ Исключен из популяции...\n")

    return margin, time_score



def sn_tests() -> float:
    count = population.count()
    bound = seq.minimum_bounding_n(config.P_VALUE / count)
    max_score = population.max_scores() or bound

    return max(1, bound + (count - max_score))



def sn_step(hunter: population.Organism) -> Optional[population.Organism]:
    """Один шаг эволюции."""
    skip = True

    if not hunter.scores or hunter.date == _end:
        skip = False

    label = ""
    if not hunter.scores:
        label = " - новый организм"

    print(f"!!! Skip={skip}   Родитель{label}:")
    if (margin := sn_eval_organism(hunter)) is None:
        print(f"!!!!1 margin = {margin}")
        return None
    print(f"!!!!2 margin = {margin}")
    if margin[0] < 0:
        return None
    if skip:
        return None
#    if (rnd := np.random.random()) < (slowness := margin[1]):
#        print(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")
#        return None

    for n_child in itertools.count(1):
        print(f"Потомок {n_child} (Scale={sn_scale():.2f}):")

        hunter = hunter.make_child(1 / sn_scale())
        if (margin := sn_eval_organism(hunter)) is None:
            return None
        if margin[0] < 0:
            return None
#        if (rnd := np.random.random()) < (slowness := margin[1]):
#            print(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")
#            return None


def sn_eval_organism(organism: population.Organism) -> Optional[tuple[float, float]]:
    """Оценка организмов.

    - Если организм уже оценен для данной даты, то он не оценивается.
    - Если организм старый, то оценивается один раз.
    - Если организм новый, то он оценивается для определенного количества дат из истории.
    """
    try:
        print(f"{organism}\n")
    except AttributeError as err:
        organism.die()
        print(f"Удаляю - {err}\n")

        return None

    all_dates = listing.all_history_date(_tickers, end=_end)
#    dates = all_dates[-sn_tests :].tolist()
    tmpsn = sn_tests()        				# кол-во тестов.
    dates = all_dates[-tmpsn :].tolist()		# торговые даты от самой старой (слева) до самой новой (справа).  На кол-во тестов

    print(f"!!!!!!!!!!!! tmpsn={tmpsn}, dates={dates}, _end={_end}, organism.date={organism.date}, organism.scores={organism.scores}, population.max_scores={population.max_scores()}")
#   _end   # - (последний торговый день (ближайшая к текущей торговая дата назад)
#   organism.date  # Последняя дата тестирования организма (он тестируется от старой даты к текущей)
#   organism.scores   # Сколько тестов прошел организм.
#   population.max_scores()   # максимальный wins популяции

    if organism.date == _end and (sn_tests() < population.max_scores() or organism.scores == sn_tests() - 1):	## На один день в прошлое
        dates = [all_dates[-(organism.scores + 1)]]
        print(f"!!!!!!!!!!!! 11111")
    elif organism.date == _end and sn_tests() >= population.max_scores() and organism.scores < sn_tests() - 1:
        organism.clear()
        print(f"!!!!!!!!!!!! 22222")
    elif organism.scores:              # Если новый алгоритм - у него пустой scores. Оценивает диапазон на текущую дату.   Остановленный на этапе тестирования организм    начинает отсюда и возможно ломает
        dates = [_end]
        print(f"!!!!!!!!!!!! 33333")

    for date in dates:
        print(f"!!!!!!!!!!!! date={date}  in  dates={dates}")
        try:
            organism.evaluate_fitness(_tickers, date)
        except (ModelError, AttributeError) as error:
            organism.die()
            print(f"Удаляю - {error}\n")

            return None

    return sn_get_margin(organism)
#    return None

import pandas as pd
from poptimizer.shared import col


def LoadFromTV(ticker: str, TVformula: str, mult: int, sn_dates: list):
#def LoadFromTV(ticker: str, TVformula: str, mult: int, gazp: pd.DataFrame):
## Тут загружаем в базу quotes все нужные данные для тех бумаг, готорый нет в TQBR

        sn_ticker = ticker
        sn_df_quotes = pd.DataFrame(columns = [col.DATE, col.OPEN, col.CLOSE, col.HIGH, col.LOW, col.TURNOVER])
        sn_df_quotes = sn_df_quotes.set_index(col.DATE)

#        import datetime
#        bars = 10
#        datetime.datetime.strptime(last_date, "%Y-%m-%d").date() - datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

        from tvDatafeed import TvDatafeed,Interval
        tv = TvDatafeed()
        tv.clear_cache()
#        t_GLDRUB = tv.get_hist("MOEX:GLDRUB_TOM", interval=Interval.in_daily,n_bars=bars.days)
        t_GLDRUB = tv.get_hist(TVformula, interval=Interval.in_daily,n_bars=10000)
#        self._logger(f"SNLOG_09. TV={t_GLDRUB}")

        sn_df_quotes[col.OPEN] = t_GLDRUB['open']
        sn_df_quotes[col.CLOSE] = t_GLDRUB['close']
        sn_df_quotes[col.HIGH] = t_GLDRUB['high']
        sn_df_quotes[col.LOW] = t_GLDRUB['low']
        sn_df_quotes[col.TURNOVER] = t_GLDRUB['volume'].mul(t_GLDRUB['close']) * mult
        sn_df_quotes.index = sn_df_quotes.index.normalize()         # Откинем часы


#  Оставим только те даты, за которые котировки уже есть
        sn_df_quotes = sn_df_quotes.loc[sn_df_quotes.index.isin(  sn_dates  )]

#  Сохраним новую котировку
        from pymongo import MongoClient
        import pickle

        client = MongoClient('localhost', 27017)
        db = client['data']
        quotes_collection = db['quotes']

        import datetime
        post = {"_id": ticker,
          "data": sn_df_quotes.to_dict("split"),
          "timestamp": datetime.datetime.utcnow()}

        quotes_collection.replace_one(filter={"_id": ticker}, replacement=post, upsert=True)


        import time
        time.sleep(10)
#        exit()






#  загрузим список тикеров и дат, т.к. другие библиотеки poptimizer не работают
from pymongo import MongoClient
quotes_collection = MongoClient('localhost', 27017)['data']['quotes']
sn_tickers = []   # список
sn_dates = []   # список
for quote in quotes_collection.find():
    sn_tickers.append(quote['_id'])

    sn_min = min(quote['data']['index'], default="EMPTY")
    sn_max = max(quote['data']['index'], default="EMPTY")
    print(f"minmax {quote['_id']} {sn_min} {sn_max}")

    for one_date in quote['data']['index']:
        if one_date not in sn_dates:
            sn_dates.append(one_date)
dates.sort()
print(sn_dates)

#LoadFromTV('GLDRUB_TOM', "MOEX:GLDRUB_TOM", 1, sn_dates)
quit()












#  загрузим список тикеров и дат, т.к. другие библиотеки poptimizer не работают
from pymongo import MongoClient
quotes_collection = MongoClient('localhost', 27017)['data']['quotes']
sn_tickers = []   # список
sn_dates = []   # список
for quote in quotes_collection.find():
    sn_tickers.append(quote['_id'])
    for one_date in quote['data']['index']:
        if one_date not in sn_dates:
            sn_dates.append(one_date)
sn_dates.sort()


#LoadFromTV('GLDRUB_TOM', "MOEX:GLDRUB_TOM", 1, sn_dates)
#LoadFromTV('SLVRUB_TOM', "MOEX:SLVRUB_TOM", 1, sn_dates)
#LoadFromTV('BTCRUB', "BINANCE:BTCRUB/10000", 10000, sn_dates)
#LoadFromTV('ETHRUB', "BINANCE:ETHRUB/100", 100, sn_dates)

quit()


from pymongo import MongoClient
quotes_collection = MongoClient('localhost', 27017)['data']['quotes']
sn_tickers = []   # список
sn_dates = []   # список
for quote in quotes_collection.find():
    sn_tickers.append(quote['_id'])
#    print(quote['data']['index'])
    for one_date in quote['data']['index']:
        if one_date not in sn_dates:
            sn_dates.append(one_date)
sn_dates.sort()

print(sn_dates)
quit()
print(sn_tickers)






quit()






from time import gmtime, strftime, localtime
#print("")
#print("Current Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))   #gmtime()))
#print("")

import pandas as pd
from poptimizer.shared import col


sn_ticker = 'BTCRUB'
sn_df_quotes = pd.DataFrame(columns = [col.DATE, col.OPEN, col.CLOSE, col.HIGH, col.LOW, col.TURNOVER])
sn_df_quotes = sn_df_quotes.set_index(col.DATE)


#            LoadFromTV('SLVRUB_TOM', "MOEX:SLVRUB_TOM", 1, 'GAZP')
#            LoadFromTV('BTCRUB', "BINANCE:BTCRUB/10000", 10000, 'GAZP')



from tvDatafeed import TvDatafeed,Interval
tv = TvDatafeed()
tv.clear_cache()
t_GLDRUB = tv.get_hist("BINANCE:BTCRUB/10000", interval=Interval.in_daily,n_bars=10000)

sn_df_quotes[col.OPEN] = t_GLDRUB['open']
sn_df_quotes[col.CLOSE] = t_GLDRUB['close']
sn_df_quotes[col.HIGH] = t_GLDRUB['high']
sn_df_quotes[col.LOW] = t_GLDRUB['low']
sn_df_quotes[col.TURNOVER] = t_GLDRUB['volume'].mul(t_GLDRUB['close']) * 10000
sn_df_quotes.index = sn_df_quotes.index.normalize()

tickers = load_tickers()
sn_df_quotes = sn_df_quotes.loc[sn_df_quotes.index.isin(  listing.all_history_date(tickers)  )]
print(sn_df_quotes.tail(10))


quit()


sn_df_quotes.drop(labels = listing.all_history_date(tickers),
    axis = 0,
    inplace = True
)

#print(sn_df_quotes)


quit()



tickers = load_tickers()
print(listing.all_history_date(tickers))
quit()

#for ticker in tickers:
#print(listing.all_history_date(tickers,start=localtime(),end=localtime()))

#from poptimizer.data.app import viewers
#from poptimizer.data import ports

import pandas as pd
from poptimizer.data.views import quotes

#print(quotes.prices(("AKRN", "GMKN", "MSTT"), pd.Timestamp("2022-10-16")))
for ticker in tickers:
    print(quotes.prices((ticker, ticker), pd.Timestamp("2022-10-16")))
#for oneprice in quotes.prices(("AKRN", "GMKN", "MSTT"), pd.Timestamp("2022-10-16")):
#    print(oneprice)
#    print(oneprice.index, oneprice.values)

#print(listing.all_history_date(tickers,start='2022-10-10',end='2022-10-10'))
#    print(ticker)
quit()






##
collection = store.get_collection()
#collection.delete_one("6344b66a1a85833dfcdfde5a")
for org in population.get_all():
#    print(org.id)
    if str(org.id) in (
"6344b66a1a85833dfcdfde5a",
"6344b8411a85833dfcdfde5c",
"63457c78636053dd92556ddf",
"63485781ac2afce6962bf25c",
"634933ffac2afce6962bf326",
"6349360bac2afce6962bf328",
"6349485fa5657a22c93bcb15",
"6349686ca5657a22c93bcb2d",
"634969e9a5657a22c93bcb2e",
"63496ea9a5657a22c93bcb31",
"634987fba5657a22c93bcb41",
"634988ada5657a22c93bcb42",
"6349b760a5657a22c93bcb5f",
"6349e69170b6e7de108df9cb",
"6349f6d170b6e7de108df9d6",
"6349fafa70b6e7de108df9d8",
"634a03ee70b6e7de108df9df",
"634a049c70b6e7de108df9e0",
"634a082e70b6e7de108df9e4",
"634a2dd970b6e7de108df9f2",
"634a3db570b6e7de108df9fc",
"634a42d270b6e7de108dfa01",
"634a45e570b6e7de108dfa03",
"632d0e1b1bd8434c1e6134d0",
"634504311a85833dfcdfde80",
"634508951a85833dfcdfde82",
"63451a07a6c270cff621675f",
"63451b52a6c270cff6216760",
"6348fff0ac2afce6962bf2f6",
"63490696ac2afce6962bf2fe",
"634937f8ac2afce6962bf32a",
"634949ffa5657a22c93bcb17",
"63494b83a5657a22c93bcb19",
"63495806a5657a22c93bcb21",
"6349622da5657a22c93bcb27",
"6349ad7ca5657a22c93bcb59",
"6349b859a5657a22c93bcb60",
"6349de8170b6e7de108df9c3",
"634a45e570b6e7de108dfa03",

"632bb1bbf301bb5b84ca7adf",
"633c3a6ee8249177d2e3eb0d",
"63456743636053dd92556dd4",
"634586e0636053dd92556de4",
"6346910fdeba86caeb191f30",
"63475414ac2afce6962bf134",
"63475ce4ac2afce6962bf143",
"6347b345ac2afce6962bf1a9",
"6348bcc1ac2afce6962bf2c7",
"6348dcd5ac2afce6962bf2d4",
"6348de83ac2afce6962bf2d6",
"6348ea30ac2afce6962bf2e2",
"6349874ea5657a22c93bcb40",

"632da1351bd8434c1e6134f7",
"632dc99d1bd8434c1e613502",
"632dcdff1bd8434c1e613503",
"632e3fa31bd8434c1e613522",
"632e426d1bd8434c1e613523",
"632e6c3f1bd8434c1e61352f",
"632e6eb21bd8434c1e613530",
"632ee3b31bd8434c1e61354d",
"632ee64e1bd8434c1e61354e",
"632f12f31bd8434c1e613559",
"632f16271bd8434c1e61355a",
"632fc316fbc5e72ca27991a9",
"633392edc60722cd4926e96c",
"6334af25c60722cd4926e98b",
"6334b368c60722cd4926e98c",
"633525ecc60722cd4926e99f",
"63352a9ec60722cd4926e9a0",
"633bc3d6e8249177d2e3eafd",
"633ce1ac6ea1d94a4552b84b",

## Это deleted_6   Зря его сделал, т.к. LLH сильно снизилось

#"632ad2429dbf8ba893b5e25a",
#"632c1e0ff301bb5b84ca7af6",
#"632c81d5e7daf3732b117ded",
#"632ca3f21bd8434c1e6134ba",
#"632d3d6c1bd8434c1e6134db",
#"6344a4741a85833dfcdfde4a",
#"6344a55b1a85833dfcdfde4b",
#"6344a9291a85833dfcdfde4f",
#"6344ab3c1a85833dfcdfde51",
#"6344ac8c1a85833dfcdfde52",
#"6344b0291a85833dfcdfde55",
#"6344b1311a85833dfcdfde56",
#"6344b3ee1a85833dfcdfde58",
#"6344bc571a85833dfcdfde5f",
#"6344bf441a85833dfcdfde61",
#"6344c1af1a85833dfcdfde63",
#"6344c4181a85833dfcdfde66",
#"6344d8e31a85833dfcdfde6d",
#"6344da931a85833dfcdfde6e",
#"6344dc4f1a85833dfcdfde6f",
#"6344eb8c1a85833dfcdfde73",
#"6344f7751a85833dfcdfde7b",
#"6344f8e01a85833dfcdfde7c",
#"634545ea636053dd92556dc2",
#"634549b5636053dd92556dc4",
#"634556ed636053dd92556dca",
#"634568eb636053dd92556dd5",
#"63456af8636053dd92556dd7",
#"63457e10636053dd92556de0",
#"63458eeb636053dd92556de8",
#"634592e0636053dd92556dea",
#"6345c091636053dd92556df8",
#"6345d2ff636053dd92556e00",




):
        print(org.id, "Delete")
        org.die()


####



_tickers = None
_end = None


step = 1
org = None
#_setup()


#step = _step_setup(step)   ############:
d_min, d_max = population.min_max_date()
if _tickers is None:
    _tickers = load_tickers()
    _end = d_max or listing.all_history_date(_tickers)[-1]

dates = listing.all_history_date(_tickers, start=_end)
if (d_min != _end) or (len(dates) == 1):
    step = step + 0
#    print(step + 1)

else:
    _end = dates[1]

#print(_tickers, _end, step)
##################


##org = population.get_next_one()

for org in population.get_all():
    if 1==2 and str(org.id) in (
#"6333e6c8c60722cd4926e973"   #its good!
#"63491b11ac2afce6962bf314"
"634a79f47df062bb7803da0e"
):
        print("Working with ", org.id)
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.
        sn_step(org)				# Обучить эту модель на один день.

        quit()



for org in population.get_all():
    pickled_model = org._doc.model

    import io
    import torch


    buffer = io.BytesIO(pickled_model)
    state_dict = torch.load(buffer)
#   print(state_dict)

#    import dumper
#    dumper.max_depth = 50
#    dumper.instance_dump = 'all'
#    dumper.dump(state_dict)

    import pickle
#    collection = Collection(MONGO_CLIENT[DB]["sn_speedy"])
    collection_speedy = snspeedy_get_collection()
#    collection_speedy = Collection(MONGO_CLIENT[DB]["sn_speedy"])

#save
#    import bson
#    speedy_id = collection_speedy.insert_one({
#        "bin_var1": bson.Binary(pickle.dumps(state_dict)),
#        "bin_var2": bson.Binary(pickle.dumps(state_dict)),
#    })

#restorev
#    record = collection_speedy.find_one({'_id': speedy_id.inserted_id})
#    restored = pickle.loads(record["bin_var1"])
#    restored = pickle.loads(record["bin_var2"])

#    dumper.dump(restored)

#clear
#    collection_speedy.delete_one({'_id': speedy_id.inserted_id})

#    quit()


##

collection = store.get_collection()
db_find = collection.find




p_rets = sn_get_keys("ir")
p_llhs = sn_get_keys("llh")
p_timers = sn_get_keys("timer")
p_wins = sn_get_keys("wins")
p_history_days = sn_get_keys("genotype", "Data", "history_days")
p_epochs = sn_get_keys("genotype", "Scheduler", "epochs")
p_tickers = sn_get_keys("tickers")
p_date = sn_get_keys("Date")   ########!!!!!!!!!!!!!!!!!!!!

p_tickers_cnt = []
for i in range(0, len(p_tickers)):
    p_tickers_cnt.append((p_tickers[i][0], len(p_tickers[i][1])))


## margin calc  to manual clear   HERE
p_metrics = []
p_timer_delta = []

for org in population.get_all():
    margin = np.inf

    LLH_median, LLH_upper, LLH_maximum = _select_worst_bound(candidate={"date": org.date, "llh": org.llh, "ir": org.ir}, metric="llh")
    valid = LLH_upper != LLH_median
    margin = min(margin, valid and (LLH_upper / (LLH_upper - LLH_median)))
    LLH_margin = margin

    RET_median, RET_upper, RET_maximum = _select_worst_bound(candidate={"date": org.date, "llh": org.llh, "ir": org.ir}, metric="ir")
    valid = RET_upper != RET_median
    margin = min(margin, valid and (RET_upper / (RET_upper - RET_median)))
    RET_margin = valid and (RET_upper / (RET_upper - RET_median))

    if margin == np.inf:
        margin = 0

    p_metrics.append([org.id, (margin, RET_margin, RET_median, RET_upper, RET_maximum, LLH_margin, LLH_median, LLH_upper, LLH_maximum)])
#    print(org.id, margin, RET_margin, RET_median, RET_upper, RET_maximum, LLH_margin, LLH_median, LLH_upper, LLH_maximum)   #0.4f

    p_timer_delta.append([org.id, (sn_time_delta(org))])




#RET worst !!!!
#upper inf  !!!



#timer in proc
#tickers in pickle

#auto clear recommendation

# удалять по ID
# Проверить и импортировать лучшие из моделей Михаила и мои старые

# В телегу!

# средний RET|LLH
from time import gmtime, strftime, localtime
print("")
print("Current Time: ", strftime("%Y-%m-%d %H:%M:%S", localtime()))   #gmtime()))
print("")




dates = listing.all_history_date(_tickers)
ddt_str =  ""
for ddt in dates[len(dates)-160:] :
    import date_converter
    converted_date = date_converter.string_to_string(str(ddt), '%Y-%m-%d %H:%M:%S', '%Y-%m-%d')
    ddt_str = f"{converted_date}\t{ddt_str}" 
#print (ddt_str)
#quit()




#print("Num\tID\tcrDate\tcrTime\tRET\tLLH\tWins\tTimer\tEpochs\thDays\tticker_cnt")
print(
    "\t".join(
        [
            f"Num",
            f"ID",
            f"crDate",
            f"crTime",
            f"RET",
            f"LLH",
            f"Wins",
            f"Timer",
            f"TimRnk",
            f"Epch",
            f"Days",
            f"tckr",
            f"margin",
            f"RET_mar",
            f"RET_med",
            f"RET_upp",
            f"RET_max",
            f"LLH_mar",
            f"LLH_med",
            f"LLH_upp",
            f"LLH_max",
            ddt_str,
        ],
    ),
)

#print(
#    "\t".join(
#        [
#            f"Num",
#            f"ID",
#            f"crDate",
#            f"crTime",
#            f"",
#            f"",
#            f"",
#            f"",
#            f"",
#            f"allRETs",
#        ],
#    ),
#)

#print(
#    "\t".join(
#        [
#            f"Num",
#            f"ID",
#            f"crDate",
#            f"crTime",
#            f"",
#            f"",
#            f"",
#            f"",
#            f"",
#            f"allLLHs",
#        ],
#    ),
#)

for i in range(0, len(p_wins)):

    import datetime as dt
    seconds_since_unix_epoch = int(p_wins[i][0][:8], base=16) + 3*60*60
    crDate = dt.datetime.utcfromtimestamp(seconds_since_unix_epoch).strftime('%Y-%m-%d')
    crTime = dt.datetime.utcfromtimestamp(seconds_since_unix_epoch).strftime('%H:%M:%S')

    quantiles = np.quantile(p_rets[i][1], [0, 0.5, 1.0])
    quantiles = map(lambda quantile: f"{quantile:.2f}", quantiles)
    quantiles_r = tuple(quantiles)[1]

    quantiles = np.quantile(p_llhs[i][1], [0, 0.5, 1.0])
    quantiles = map(lambda quantile: f"{quantile:.2f}", quantiles)
    quantiles_l = tuple(quantiles)[1]

    print(
        "\t".join(
            [
                f"{i+1}",
                f"{p_wins[i][0]}",
                f"{crDate}",
                f"{crTime}",
                f"{quantiles_r}",
                f"{quantiles_l}",
                f"{p_wins[i][1]}",
                f"{p_timers[i][1]}",
                f"{p_timer_delta[i][1]:.2f}",
                f"{p_epochs[i][1]:.2f}",
                f"{p_history_days[i][1]:.2f}",
                f"{p_tickers_cnt[i][1]}",
                f"{p_metrics[i][1][0]:.2f}",
                f"{p_metrics[i][1][1]:.2f}",
                f"{p_metrics[i][1][2]:.2f}",
                f"{p_metrics[i][1][3]:.2f}",
                f"{p_metrics[i][1][4]:.2f}",
                f"{p_metrics[i][1][5]:.2f}",
                f"{p_metrics[i][1][6]:.2f}",
                f"{p_metrics[i][1][7]:.2f}",
                f"{p_metrics[i][1][8]:.2f}",
            ],
        ),
    )

    tmpstr = ""
    for v in p_rets[i][1]:
        tmpstr = tmpstr + "\t" + "{:.2f}".format(v)
#    print(f"{i+1}\t{p_rets[i][0]}\t{crDate}\t{crTime}\tallRETs\t\t\t\t\t\t{tmpstr}")
    print(
        "\t".join(
            [
                f"{i+1}",
                f"{p_rets[i][0]}",
                f"{crDate}",
                f"{crTime}",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"allRETs",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"{tmpstr}",
            ],
        ),
    )


    tmpstr = ""
    for v in p_llhs[i][1]:
        tmpstr = tmpstr + "\t" + "{:.2f}".format(v)
#    print(f"{i+1}\t{p_rets[i][0]}\t{crDate}\t{crTime}\tallLLHs\t\t\t\t\t\t{tmpstr}")
    print(
        "\t".join(
            [
                f"{i+1}",
                f"{p_rets[i][0]}",
                f"{crDate}",
                f"{crTime}",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"allLLHs",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"{tmpstr}",
            ],
        ),
    )


#    time_score = _time_delta(org)
#    self._logger.info(f"Margin - {margin:.2%}, Slowness - {time_score:.2%}\n")  # noqa: WPS221






#quantiles = np.quantile(p_rets[i][1], [0, 0.5, 1.0])
#quantiles = map(lambda quantile: f"{quantile:.4f}", quantiles)
#quantiles = tuple(quantiles)
#print(quantiles)
#print(quantiles[1])
#quantiles = ", ".join(tuple(quantiles))



#keys = map(lambda doc: doc[key], cursor)
#LOGGER.info(f"{keys}")

#keys = map(
#    lambda amount: amount if isinstance(amount, float) else np.median(np.array(amount)),
#    keys,
#)
#keys = filter(
#    lambda amount: not np.isnan(amount),
#    keys,
#)
#keys = tuple(keys)

#if keys:
#    quantiles = np.quantile(keys, [0, 0.5, 1.0])
#    quantiles = map(lambda quantile: f"{quantile:.4f}", quantiles)
#    quantiles = tuple(quantiles)
#else:
#    quantiles = ["-" for _ in range(3)]

#quantiles = ", ".join(tuple(quantiles))
#view = view or key.upper()

#LOGGER.info(f"{view} - ({quantiles})")  # noqa: WPS421



quit()


_START_POPULATION: Final = 100


class Evolution:  # noqa: WPS214
    """Эволюция параметров модели.

    Эволюция состоит из бесконечного создания организмов и сравнения их характеристик с медианными значениями по
    популяции. Сравнение осуществляется с помощью последовательного теста для медиан, который учитывает изменение
    значимости тестов при множественном тестировании по мере появления данных за очередной период времени. Дополнительно
    осуществляется коррекция на множественное тестирование на разницу llh и доходности.
    """

    def __init__(self):
        """Инициализирует необходимые параметры."""
        self._tickers = None
        self._end = None
        self._logger = logging.getLogger()

    @property
    def _scale(self) -> float:
        return population.count() ** 0.5

    @property
    def _tests(self) -> float:
        count = population.count()
        bound = seq.minimum_bounding_n(config.P_VALUE / count)
        max_score = population.max_scores() or bound

        return max(1, bound + (count - max_score))

    def evolve(self) -> None:
        """Осуществляет эволюции.

        При необходимости создается начальная популяция из случайных организмов по умолчанию.
        """
        step = 0
        org = None

        self._setup()

        while _check_time_range(self):
            step = self._step_setup(step)

            date = self._end.date()
            self._logger.info(f"***{date}: Шаг эволюции — {step}***")
            population.print_stat()
            self._logger.info(f"Тестов - {self._tests}\n")

            if org is None:
                org = population.get_next_one()

            org = self._step(org)

    def _step_setup(
        self,
        step: int,
    ) -> int:
        d_min, d_max = population.min_max_date()
        if self._tickers is None:
            self._tickers = load_tickers()
            self._end = d_max or listing.all_history_date(self._tickers)[-1]

        dates = listing.all_history_date(self._tickers, start=self._end)
        if (d_min != self._end) or (len(dates) == 1):
            return step + 1

        self._end = dates[1]

        return 1

    def _setup(self) -> None:
        if population.count() == 0:
            for i in range(1, _START_POPULATION + 1):
                self._logger.info(f"Создается базовый организм {i}:")
                org = population.create_new_organism()
                self._logger.info(f"{org}\n")

    def _step(self, hunter: population.Organism) -> Optional[population.Organism]:
        """Один шаг эволюции."""
        skip = True

        if not hunter.scores or hunter.date == self._end:
            skip = False

        label = ""
        if not hunter.scores:
            label = " - новый организм"

        self._logger.info(f"Родитель{label}:")
        if (margin := self._eval_organism(hunter)) is None:
            return None
        if margin[0] < 0:
            return None
        if skip:
            return None
        if (rnd := np.random.random()) < (slowness := margin[1]):
            self._logger.info(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")

            return None

        for n_child in itertools.count(1):
            self._logger.info(f"Потомок {n_child} (Scale={self._scale:.2f}):")

            hunter = hunter.make_child(1 / self._scale)
            if (margin := self._eval_organism(hunter)) is None:
                return None
            if margin[0] < 0:
                return None
            if (rnd := np.random.random()) < (slowness := margin[1]):
                self._logger.info(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")

                return None

    def _eval_organism(self, organism: population.Organism) -> Optional[tuple[float, float]]:
        """Оценка организмов.

        - Если организм уже оценен для данной даты, то он не оценивается.
        - Если организм старый, то оценивается один раз.
        - Если организм новый, то он оценивается для определенного количества дат из истории.
        """
        try:
            self._logger.info(f"{organism}\n")
        except AttributeError as err:
            organism.die()
            self._logger.error(f"Удаляю - {err}\n")

            return None

        all_dates = listing.all_history_date(self._tickers, end=self._end)
        dates = all_dates[-self._tests :].tolist()

        if organism.date == self._end and (self._tests < population.max_scores() or organism.scores == self._tests - 1):
            dates = [all_dates[-(organism.scores + 1)]]
        elif organism.date == self._end and self._tests >= population.max_scores() and organism.scores < self._tests - 1:
            organism.clear()
        elif organism.scores:
            dates = [self._end]

        for date in dates:
            try:
                organism.evaluate_fitness(self._tickers, date)
            except (ModelError, AttributeError) as error:
                organism.die()
                self._logger.error(f"Удаляю - {error}\n")

                return None

        return self._get_margin(organism)

    def _get_margin(self, org: population.Organism) -> tuple[float, float]:
        """Используется тестирование разницы llh и ret против самого старого организма.

        Используются тесты для связанных выборок, поэтому предварительно происходит выравнивание по
        датам и отбрасывание значений не имеющих пары (возможно первое значение и хвост из старых
        значений более старого организма).
        """
        margin = np.inf

        names = {"llh": "LLH", "ir": "RET"}

        for metric in ("llh", "ir"):
            median, upper, maximum = _select_worst_bound(
                candidate={"date": org.date, "llh": org.llh, "ir": org.ir},
                metric=metric,
            )

            self._logger.info(
                " ".join(
                    [
                        f"{names[metric]} worst difference:",
                        f"median - {median:0.4f},",
                        f"upper - {upper:0.4f},",
                        f"max - {maximum:0.4f}",
                    ],
                ),
            )

            valid = upper != median
            margin = min(margin, valid and (upper / (upper - median)))

        if margin == np.inf:
            margin = 0

        time_score = _time_delta(org)

        self._logger.info(f"Margin - {margin:.2%}, Slowness - {time_score:.2%}\n")  # noqa: WPS221

        if margin < 0:
            org.die()
            self._logger.info("Исключен из популяции...\n")

        return margin, time_score


def _time_delta(org):
    """Штраф за время, если организм медленнее медианного в популяции."""
    times = [doc["timer"] for doc in population.get_metrics()]

    return stats.percentileofscore(times, org.timer, kind="mean") / 100


def _check_time_range(self) -> bool:
## SNADDED

    import os

    directory = '/home/sn/sn/poptimizer-master/auto/!work'
    files = os.listdir(directory)
    bashes = filter(lambda x: x.endswith('.sh'), files)

    for bash in sorted(bashes):
         self._logger.info(f'Running {directory}/{bash}')
         os.system(f'{directory}/{bash} 2>&1 | tee {directory}/{bash}_out')
#         os.system(f'{directory}/{bash} > {directory}/{bash}_out')
         os.system(f'rm {directory}/{bash}')
#         quit()


###########



    hour = datetime.datetime.today().hour

    if config.START_EVOLVE_HOUR == config.STOP_EVOLVE_HOUR:
        return True

    if config.START_EVOLVE_HOUR < config.STOP_EVOLVE_HOUR:
        return config.START_EVOLVE_HOUR <= hour < config.STOP_EVOLVE_HOUR

    before_midnight = config.START_EVOLVE_HOUR <= hour
    after_midnight = hour < config.STOP_EVOLVE_HOUR

    return before_midnight or after_midnight


def _select_worst_bound(candidate: dict, metric: str) -> tuple[float, float, float]:
    """Выбирает минимальное значение верхней границы доверительного интервала.

    Если данный организм не уступает целевому организму, то верхняя граница будет положительной.
    """

    diff = _aligned_diff(candidate, metric)

    bounds = map(
        lambda size: _test_diff(diff[:size]),
        range(1, len(diff) + 1),
    )

    return min(
        bounds,
        key=lambda bound: bound[1] or np.inf,
    )


def _aligned_diff(candidate: dict, metric: str) -> list[float]:
    comp = []

    for base in population.get_metrics():
        metrics = base[metric]

        if base["date"] < candidate["date"]:
            metrics = [np.nan] + metrics

        scores = len(candidate[metric])

        metrics = metrics[:scores]
        metrics = metrics + [np.nan] * (scores - len(metrics))

        comp.append(metrics)

    comp = np.nanmedian(np.array(comp), axis=0)

    return list(map(operator.sub, candidate[metric], comp))


def _test_diff(diff: list[float]) -> tuple[float, float, float]:
    """Последовательный тест на медианную разницу с учетом множественного тестирования.

    Тестирование одностороннее, поэтому p-value нужно умножить на 2, но проводится 2 раза.
    """
    _, upper = seq.median_conf_bound(diff, config.P_VALUE / population.count())

    return float(np.median(diff)), upper, np.max(diff)
