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
#from poptimizer.data.views import listing
from poptimizer.dl import ModelError
from poptimizer.evolve import population, seq
from poptimizer.portfolio.portfolio import load_tickers
from poptimizer.evolve import store
import logging
from pymongo.collection import Collection
from poptimizer.store.database import DB, MONGO_CLIENT
from poptimizer.data.views import quotes
import pandas as pd

population.print_stat()

LOGGER = logging.getLogger()

_COLLECTION = MONGO_CLIENT[DB]["sn_speedy"]
def snspeedy_get_collection() -> Collection:
    return _COLLECTION

def all_history_date(tickers: tuple[str, ...], *, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.Index:
    """Перечень дат для которых есть котировки после проверки на наличие новых данных.

    Может быть ограничен сверху или снизу.
    """
    return quotes.all_prices(tickers).loc[start:end].index

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

def sn_get_attr_from_list(objid: population.Organism.id, attrlist: list, s_type: str = '') -> Optional[any]:
    for attr in attrlist:
        if str(attr[0]) == str(objid):
#            print(attr[0], objid)
            return attr[1]
    try:
        if isinstance(attrlist[0][1], (float, int)):
            return 0
        elif isinstance(attrlist[0][1], (str)):
            return ''
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    except:
        if s_type == 's':
            return ''
        elif s_type == 'f':
            return 0
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]


"""
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
"""




#######################################################################################
collection = store.get_collection()


"""

#collection.delete_one("6344b66a1a85833dfcdfde5a")
for org in population.get_all():
#    print(org.id)
    if str(org.id) in (

"637da896ed20ae4c0f4b142e",
"63858df58ad4aea51d849864",
"638dbacf62f78314766d9531",
"6396efe00941f8f33c0851de",

):
        print(org.id, "Delete")
        org.die()
quit()
"""


_tickers = None
_end = None


step = 1
org = None
#_setup()


#step = _step_setup(step)   ############:
d_min, d_max = population.min_max_date()
if _tickers is None:
    _tickers = load_tickers()
    _end = d_max or all_history_date(_tickers)[-1]

dates = all_history_date(_tickers, start=_end)
if (d_min != _end) or (len(dates) == 1):
    step = step + 0
#    print(step + 1)

else:
    _end = dates[1]

#print(_tickers, _end, step)
##################


"""
for org in population.get_all():
    if 1==2 and str(org.id) in (
#"6333e6c8c60722cd4926e973"   #its good!
#"63491b11ac2afce6962bf314"
#"634a79f47df062bb7803da0e"
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
"""


for org in population.get_all():
    pickled_model = org._doc.model

    import io
    import torch


    buffer = io.BytesIO(pickled_model)
    try:
        state_dict = torch.load(buffer)
#    except AttributeError as err:
    except:
        continue




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

## Проверка курсов !!!
def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list


def intersection_list(list1, list2):
   list3 = [list(filter(lambda x: x not in list1, sublist)) for sublist in list2]
   return list3

#list1 = [10, 9, 17, 40, 23, 18, 56, 49, 58, 60]
#list2 = [[25, 17, 23, 40, 32], [1, 10, 13, 27, 28], [60, 55, 61, 78, 15, 76]]
#print(intersection_list(list1, list2))
#quit()

def intersection_list_1(list1, list2):
   list3 = [list(filter(lambda x: x not in list1, list2))]
   return list3
#list1 = [10, 9, 17, 40, 23, 18, 56, 49, 58, 60]
#list2 = [25, 17, 23, 40, 32]
#print(intersection_list_1(list1, list2))
#quit()

def sn_xo(var1):
    if var1 < 0:
        return(-1)
    elif var1 > 0:
        return(1)
    else:
        return(0)



from pymongo import MongoClient
quotes_collection: object = MongoClient('localhost', 27017)['data']['quotes']
#sn_tickers = []   # список
#sn_dates = []   # список
#sn_tickers_date = []   # список

sn_gazp_dates = []   # список
sn_gazp2_dates = []   # список
sn_btcrub_dates = []   # список
sn_ethrub_dates = []   # список
sn_gldrub_tom_dates = []   # список
sn_slvrub_tom_dates = []   # список
#begdate = datetime.datetime(2022, 11, 1, 0, 0)
begdate = datetime.datetime(2020, 10, 13, 0, 0)

for quote in quotes_collection.find():
    for one_date in quote['data']['index']:
#        if quote['_id'] in ('GAZP', 'BTCRUB'):
#            sn_tickers_date.append([quote['_id'], one_date])
        if quote['_id'] == 'GAZP' and one_date >= begdate:
            sn_gazp_dates.append(one_date)
        if quote['_id'] == 'BTCRUB' and one_date >= begdate:
            sn_btcrub_dates.append(one_date)
        if quote['_id'] == 'ETHRUB' and one_date >= begdate:
            sn_ethrub_dates.append(one_date)
        if quote['_id'] == 'GLDRUB_TOM' and one_date >= begdate:
            sn_gldrub_tom_dates.append(one_date)
        if quote['_id'] == 'SLVRUB_TOM' and one_date >= begdate:
            sn_slvrub_tom_dates.append(one_date)


sn_MEOGTRR_dates = []   # список
sn_RVI_dates = []   # список
sn_IMOEX_dates = []   # список
sn_MCFTRR_dates = []   # список

begdate = datetime.datetime(2022, 11, 1, 0, 0)

indexes_collection = MongoClient('localhost', 27017)['data']['indexes']
for quote in indexes_collection.find():
    for one_date in quote['data']['index']:
        if quote['_id'] == 'GAZP' and one_date >= begdate:
            sn_gazp2_dates.append(one_date)
        if quote['_id'] == 'MEOGTRR' and one_date >= begdate:
            sn_MEOGTRR_dates.append(one_date)
        if quote['_id'] == 'RVI' and one_date >= begdate:
            sn_RVI_dates.append(one_date)
        if quote['_id'] == 'IMOEX' and one_date >= begdate:
            sn_IMOEX_dates.append(one_date)
        if quote['_id'] == 'MCFTRR' and one_date >= begdate:
            sn_MCFTRR_dates.append(one_date)




#Альтернативные есть, а GAZP нет:
#if not isListEmpty(lishka := intersection_list(sn_gazp_dates, (sn_btcrub_dates, sn_btcrub_dates, sn_gldrub_tom_dates, sn_slvrub_tom_dates ))):
#    print(f"Альтернативные есть, а GAZP нет: {lishka}")
print(f"Альтернативные котировки есть, а GAZP нет: {lishka}") if not isListEmpty(lishka := intersection_list(sn_gazp_dates, (sn_btcrub_dates, sn_btcrub_dates, sn_gldrub_tom_dates, sn_slvrub_tom_dates ))) else ""
print(f"Альтернативные индексы есть, а GAZP нет: {lishka}") if not isListEmpty(lishka := intersection_list(sn_gazp_dates, (  sn_MEOGTRR_dates, sn_RVI_dates, sn_IMOEX_dates, sn_MCFTRR_dates  ))) else ""



#GAZP есть, а Альтернативных нет:
#if not isListEmpty(lishka := intersection_list_1(  sn_btcrub_dates, sn_gazp_dates   )):
#    print(f"btcrub есть, а GAZP нет: {lishka}")
print(f"GAZP есть, а btcrub нет: {lishka}")     if not isListEmpty(lishka := intersection_list_1(  sn_btcrub_dates, sn_gazp_dates       )) else ""
print(f"GAZP есть, а ethrub нет: {lishka}")     if not isListEmpty(lishka := intersection_list_1(  sn_ethrub_dates, sn_gazp_dates       )) else ""
print(f"GAZP есть, а gldrub_tom нет: {lishka}") if not isListEmpty(lishka := intersection_list_1(  sn_gldrub_tom_dates, sn_gazp_dates   )) else ""
print(f"GAZP есть, а slvrub_tom нет: {lishka}") if not isListEmpty(lishka := intersection_list_1(  sn_slvrub_tom_dates, sn_gazp_dates   )) else ""

print(f"GAZP есть, а MEOGTRR нет: {lishka}") if not isListEmpty(lishka := intersection_list_1(  sn_MEOGTRR_dates, sn_gazp2_dates   )) else ""
print(f"GAZP есть, а RVI нет: {lishka}")     if not isListEmpty(lishka := intersection_list_1(  sn_RVI_dates,     sn_gazp2_dates   )) else ""
print(f"GAZP есть, а IMOEX нет: {lishka}")   if not isListEmpty(lishka := intersection_list_1(  sn_IMOEX_dates,   sn_gazp2_dates   )) else ""
print(f"GAZP есть, а MCFTRR нет: {lishka}")  if not isListEmpty(lishka := intersection_list_1(  sn_MCFTRR_dates,  sn_gazp2_dates   )) else ""


## Конец проверки курсов


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

p_gen_batch_size = sn_get_keys("genotype", "Data", "batch_size")
p_gen_ticker_on = sn_get_keys("genotype", "Data", "ticker_on")
p_gen_day_of_year_on = sn_get_keys("genotype", "Data", "day_of_year_on")
p_gen_day_of_period_on = sn_get_keys("genotype", "Data", "day_of_period_on")
p_gen_prices_on = sn_get_keys("genotype", "Data", "prices_on")
p_gen_dividends_on = sn_get_keys("genotype", "Data", "dividends_on")
p_gen_turnover_on = sn_get_keys("genotype", "Data", "turnover_on")
p_gen_average_turnover_on = sn_get_keys("genotype", "Data", "average_turnover_on")
p_gen_rvi_on = sn_get_keys("genotype", "Data", "rvi_on")
p_gen_mcftrr_on = sn_get_keys("genotype", "Data", "mcftrr_on")
p_gen_imoex_on = sn_get_keys("genotype", "Data", "imoex_on")
p_gen_ticker_type_on = sn_get_keys("genotype", "Data", "ticker_type_on")
p_gen_usd_on = sn_get_keys("genotype", "Data", "usd_on")
p_gen_open_on = sn_get_keys("genotype", "Data", "open_on")
p_gen_high_on = sn_get_keys("genotype", "Data", "high_on")
p_gen_low_on = sn_get_keys("genotype", "Data", "low_on")
p_gen_meogtrr_on = sn_get_keys("genotype", "Data", "meogtrr_on")

p_tickers_cnt = []
for i in range(0, len(p_tickers)):
    p_tickers_cnt.append((p_tickers[i][0], len(p_tickers[i][1])))


## margin calc  to manual clear   HERE
p_metrics = []
p_timer_delta = []


for org in population.get_all():
#    print(sn_get_attr_from_list(org.id, p_history_days))
#     upper_bound = 1
# #LLH
#     margin = np.inf
#     LLH_median = LLH_upper = LLH_maximum = 0
#
#     try:
#         LLH_median, LLH_upper, LLH_maximum = _select_worst_bound(candidate={"date": org.date, "llh": org.llh, "ir": org.ir}, metric="llh")
#     except:
#         continue
#
# #    upper *= max(0, upper)
#     upper_bound *= max(0, LLH_upper)
#
#     valid = LLH_upper != LLH_median
#     margin = min(margin, valid and (LLH_upper / (LLH_upper - LLH_median)))
#     LLH_margin = margin
#
#
# #RET
#     RET_median, RET_upper, RET_maximum = _select_worst_bound(candidate={"date": org.date, "llh": org.llh, "ir": org.ir}, metric="ir")
#     upper_bound *= max(0, upper_bound, RET_upper)
#     valid = RET_upper != RET_median
#     margin = min(margin, valid and (RET_upper / (RET_upper - RET_median)))
#     RET_margin = valid and (RET_upper / (RET_upper - RET_median))
#
#     if margin == np.inf:
#         margin = 0

###
    margin = np.inf

    names = {"llh": "LLH", "ir": "RET"}

    upper_bound = 1
    LLH_median = LLH_upper = LLH_maximum = 0
    RET_median = RET_upper = RET_maximum = 0

    for metric in ("llh", "ir"):
        try:
            median, upper, maximum = _select_worst_bound(
                candidate={"date": org.date, "llh": org.llh, "ir": org.ir},
                metric=metric,
            )
        except:
            continue

        upper_bound *= max(0, upper)

        if metric == 'llh':
            LLH_median = median
            LLH_upper = upper
            LLH_maximum = maximum
        if metric == 'ir':
            RET_median = median
            RET_upper = upper
            RET_maximum = maximum

        valid = upper != median
        margin = min(margin, valid and (upper / (upper - median)))

        if metric == 'llh':
            LLH_margin = valid and (upper / (upper - median))
        if metric == 'ir':
            RET_margin = valid and (upper / (upper - median))

    upper_bound = upper_bound ** 0.5

    if margin == np.inf:
        margin = 0
###


    p_metrics.append([org.id, (margin, RET_margin, RET_median, RET_upper, RET_maximum, LLH_margin, LLH_median, LLH_upper, LLH_maximum, upper_bound)])
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




dates = all_history_date(_tickers)
ddt_str =  ""
for ddt in dates[len(dates)-250:] :
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
            f"up_bnd",
            f"RET_mar",
            f"RET_med",
            f"RET_upp",
            f"RET_max",
            f"LLH_mar",
            f"LLH_med",
            f"LLH_upp",
            f"LLH_max",


            f"batch_size",
            f"ticker_on",
            f"day_of_year_on",
            f"day_of_period_on",
            f"prices_on",
            f"dividends_on",
            f"turnover_on",
            f"average_turnover_on",
            f"rvi_on",
            f"mcftrr_on",
            f"imoex_on",
            f"ticker_type_on",
            f"usd_on",
            f"open_on",
            f"high_on",
            f"low_on",
            f"meogtrr_on",



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

i=0
for org in population.get_all():
#    print(org.id)
#    print(sn_get_attr_from_list(org.id, p_history_days))
#for i in range(0, len(p_wins)):
    i+=1

    import datetime as dt
    crDate = ''
    crTime = ''
    quantiles_r = ''
    quantiles_l = ''

    try:
#        seconds_since_unix_epoch = int(p_wins[i][0][:8], base=16) + 3*60*60
        seconds_since_unix_epoch = int(str(org.id)[:8], base=16) + 3*60*60
        crDate = dt.datetime.utcfromtimestamp(seconds_since_unix_epoch).strftime('%Y-%m-%d')
        crTime = dt.datetime.utcfromtimestamp(seconds_since_unix_epoch).strftime('%H:%M:%S')

#        quantiles = np.quantile(p_rets[i][1], [0, 0.5, 1.0])
        quantiles = np.quantile(sn_get_attr_from_list(org.id, p_rets), [0, 0.5, 1.0])
        quantiles = map(lambda quantile: f"{quantile:.2f}", quantiles)
        quantiles_r = tuple(quantiles)[1]

#        quantiles = np.quantile(p_llhs[i][1], [0, 0.5, 1.0])
        quantiles = np.quantile(sn_get_attr_from_list(org.id, p_llhs), [0, 0.5, 1.0])
        quantiles = map(lambda quantile: f"{quantile:.2f}", quantiles)
        quantiles_l = tuple(quantiles)[1]
    except:
        crDate = ''
        crTime = ''
        quantiles_r = ''
        quantiles_l = ''

    print(
        "\t".join(
            [
                f"{i}",
#sn_get_attr_from_list(org.id, p_history_days)
#                f"{p_wins[i][0]}",
                f"{org.id}",
                f"{crDate}",
                f"{crTime}",
                f"{quantiles_r}",
                f"{quantiles_l}",
#                f"{p_wins[i][1]}",
                f"{sn_get_attr_from_list(org.id, p_wins)}",
#                f"{p_timers[i][1]}",
                f"{sn_get_attr_from_list(org.id, p_timers)}",
#                f"{p_timer_delta[i][1]:.2f}",
                f"{sn_get_attr_from_list(org.id, p_timer_delta, 'f'):.2f}",
#                f"{p_epochs[i][1]:.2f}",
                f"{sn_get_attr_from_list(org.id, p_epochs):.2f}",
#                f"{p_history_days[i][1]:.2f}",
                f"{sn_get_attr_from_list(org.id, p_history_days):.2f}",
#                f"{p_tickers_cnt[i][1]}",
                f"{sn_get_attr_from_list(org.id, p_tickers_cnt)}",
#                f"{p_metrics[i][1][0]:.2f}",
#                f"{p_metrics[i][1][1]:.2f}",
#                f"{p_metrics[i][1][2]:.2f}",
#                f"{p_metrics[i][1][3]:.2f}",
#                f"{p_metrics[i][1][4]:.2f}",
#                f"{p_metrics[i][1][5]:.2f}",
#                f"{p_metrics[i][1][6]:.2f}",
#                f"{p_metrics[i][1][7]:.2f}",
#                f"{p_metrics[i][1][8]:.2f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[0]:.4f}",		# margin
                f"{sn_get_attr_from_list(org.id, p_metrics)[9]:.5f}",		# upper_bound
                f"{sn_get_attr_from_list(org.id, p_metrics)[1]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[2]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[3]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[4]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[5]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[6]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[7]:.4f}",
                f"{sn_get_attr_from_list(org.id, p_metrics)[8]:.4f}",



                f"{sn_get_attr_from_list(org.id, p_gen_batch_size):.2f}",		#other genotype
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_ticker_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_day_of_year_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_day_of_period_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_prices_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_dividends_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_turnover_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_average_turnover_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_rvi_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_mcftrr_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_imoex_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_ticker_type_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_usd_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_open_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_high_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_low_on))}",
                f"{sn_xo(sn_get_attr_from_list(org.id, p_gen_meogtrr_on))}",

            ],
        ),
    )

    tmpstr = ""
#    for v in p_rets[i][1]:
    for v in sn_get_attr_from_list(org.id, p_rets):
        tmpstr = tmpstr + "\t" + "{:.2f}".format(v)
    print(
        "\t".join(
            [
                f"{i}",
#                f"{p_rets[i][0]}",
                f"{org.id}",
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
                f"",

                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
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
#    for v in p_llhs[i][1]:
    for v in sn_get_attr_from_list(org.id, p_llhs):
        tmpstr = tmpstr + "\t" + "{:.2f}".format(v)

    print(
        "\t".join(
            [
                f"{i}",
#                f"{p_rets[i][0]}",
                f"{org.id}",
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
                f"",

                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
                f"",
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
            self._end = d_max or all_history_date(self._tickers)[-1]

        dates = all_history_date(self._tickers, start=self._end)
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

        all_dates = all_history_date(self._tickers, end=self._end)
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
