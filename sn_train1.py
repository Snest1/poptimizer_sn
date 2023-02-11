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


population.print_stat()

from poptimizer.evolve import store
import logging
LOGGER = logging.getLogger()


from poptimizer.data.views import quotes
import pandas as pd
def all_history_date(
    tickers: tuple[str, ...],
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None
) -> pd.Index:
    """Перечень дат для которых есть котировки после проверки на наличие новых данных.

    Может быть ограничен сверху или снизу.
    """
    return quotes.all_prices(tickers).loc[start:end].index



"""
def sn_alL_h_dates()
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
"""

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
#        org.die()
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


def sn_eval_organism(organism: population.Organism, toWins: int = 0) -> Optional[tuple[float, float]]:
    """Оценка организмов.

    - Если организм уже оценен для данной даты, то он не оценивается.
    - Если организм старый, то оценивается один раз.
    - Если организм новый, то он оценивается для определенного количества дат из истории.
    """
    try:
        print(f"{organism}\n")
    except AttributeError as err:
#        organism.die()
#        print(f"Удаляю - {err}\n")
        logger.info(f"!!!_                          Organizm:{organism.id}  toWins={toWins}   Проблема: 04 Удаляю(НЕ) - {err}")

        return None

#    all_dates = listing.all_history_date(_tickers, end=_end)
    all_dates = all_history_date(_tickers, end=_end)
#    dates = all_dates[-sn_tests :].tolist()
    tmpsn = sn_tests()        				# кол-во тестов.
    dates = all_dates[-tmpsn :].tolist()		# торговые даты от самой старой (слева) до самой новой (справа).  На кол-во тестов

#    print(f"!!!!!!!!!!!! tmpsn={tmpsn}, dates={dates}, _end={_end}, organism.date={organism.date}, organism.scores={organism.scores}, population.max_scores={population.max_scores()}")
#   _end   # - (последний торговый день (ближайшая к текущей торговая дата назад)
#   organism.date  # Последняя дата тестирования организма (он тестируется от старой даты к текущей)
#   organism.scores   # Сколько тестов прошел организм.
#   population.max_scores()   # максимальный wins популяции

    if toWins > 0:
        dates = (all_dates[-toWins:])
#                [all_dates[-toWins:-1]]
        print(f"!!!!!!!!!!!! toWins={toWins}, dates={dates}")
    elif organism.date == _end and (sn_tests() < population.max_scores() or organism.scores == sn_tests() - 1):	## На один день в прошлое
#        dates = [all_dates[-(organism.scores + 1):-1]]
        dates = [all_dates]
        print(f"!!!!!!!!!!!! 11111")
    elif organism.date == _end and sn_tests() >= population.max_scores() and organism.scores < sn_tests() - 1:
        organism.clear()
        print(f"!!!!!!!!!!!! 22222")
    elif organism.scores:              # Если новый алгоритм - у него пустой scores. Оценивает диапазон на текущую дату.   Остановленный на этапе тестирования организм    начинает отсюда и возможно ломает
        dates = [_end]
        print(f"!!!!!!!!!!!! 33333")

    cnt = 0
    for date in dates:
        cnt += 1
#        print(f"!!!!!!!!!!!! date={date}  in  dates={dates}")
        print(f"!!!!!!!!!!!! date={date}   {cnt} of {len(dates)}")
        try:
            organism.evaluate_fitness(_tickers, date)
        except (ModelError, AttributeError) as error:
#            organism.die()
            logger.info(f"!!!_                          Organizm:{organism.id}  toWins={toWins}   Проблема: 03 Удаляю(НЕ) - {error}")
#            print(f"Удаляю - {error}\n")

            return None

    return sn_get_margin(organism)
#    return None


def sn_step_parent(hunter: population.Organism, to_scores: int = 0, childs: int = 0, number: int = 0) -> Optional[population.Organism]:
    """Эволюция одного родителя до указанного wins и с генерацией указанного кол-ва детей"""
#    print(f"hunter.scores={hunter.scores}, to_scores={to_scores}")

    import logging
    logger = logging.getLogger()

    if hunter.scores and to_scores <= hunter.scores:
        print(f"!!! Родитель (ID={hunter.id}) уже имеет wins={hunter.scores} больше требуемого {to_scores}:")
        return None

    label = " - новый организм" if not hunter.scores else ""
    print(f"!!! Родитель{label} (ID={hunter.id}):")

    try:
        print(f"{hunter}")		# возможно не нужно его печатать.
    except AttributeError as err:
        organism.die()
        print(f"{hunter.id} Удаляю - {err}")
        logger.info(f"{hunter.id} Удаляю - {err}")
        return None

    _tickers = None
    _tickers = load_tickers()
    d_min, d_max = population.min_max_date()
#    _end = d_max or listing.all_history_date(_tickers)[-1]
    _end = d_max or all_history_date(_tickers)[-1]

#    all_dates = listing.all_history_date(_tickers, end=_end)
    all_dates = all_history_date(_tickers, end=_end)
    dates = all_dates[-to_scores :].tolist()		# торговые даты от самой старой (слева) до самой новой (справа).  На кол-во тестов

    logger.info(f"!!!!!!!!!!!! {hunter.id} dates={dates}, _end={_end}")

    for date in dates:
        logger.info(f"!!!!!!!!!!!! {hunter.id} date={date}, scores={hunter.scores} of {to_scores}. OrgNum={number}")
        try:
            hunter.evaluate_fitness(_tickers, date)
        except (ModelError, AttributeError) as error:
            hunter.die()
            print(f"{hunter.id} Удаляю - {error}\n")
            logger.info(f"{hunter.id} Удаляю - {error}")
            return None

    print(sn_get_margin(hunter))
    return hunter


### Ниже, если захотим детей еще нарожать от него.
#    print(f"!!!!1 margin = {margin}")
##    if margin[0] < 0:
##        return None
#    if skip:
#        return None
##    if (rnd := np.random.random()) < (slowness := margin[1]):
##        print(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")
##        return None
#
#    for n_child in itertools.count(1):
#        print(f"Потомок {n_child} (Scale={sn_scale():.2f}):")

#        hunter = hunter.make_child(1 / sn_scale())
#        if (margin := sn_eval_organism(hunter)) is None:
#            return None
#        if margin[0] < 0:
#            return None
##        if (rnd := np.random.random()) < (slowness := margin[1]):
##            print(f"Медленный не размножается {rnd=:.2%} < {slowness=:.2%}...\n")
##            return None


def sn_get_org_by_id(str_org: str) -> population.Organism:
    for org in population.get_all():
        if str(org.id) == str_org:
            genotype = org.genotype
            return org

from poptimizer.evolve.genotype import Genotype



def sn_make_core_child(par1: population.Organism = None, par2: population.Organism = None, scale: float = 0) -> "Organism":
    """Создает новый организм с помощью дифференциальной мутации."""
#    parent1, parent2 = _get_parents()
    genotype=par1.genotype
    parent1 = par1
    parent2 = par2
    child_genotype = genotype.make_child(parent1.genotype, parent2.genotype, scale)
#    child_genotype = sn_geno_make_child(parent1.genotype, parent2.genotype, scale)

    return population.Organism(genotype=child_genotype)


def sn_make_child(hunter: population.Organism, toWins: int, scale: float = 0, check_margin: bool = False) -> Optional[population.Organism]:
    """Одного ребенка родить"""

    logger.info(f"!!! Родитель: {hunter.id}  toWins={toWins}")

#    child = hunter.make_child(1 / sn_scale())
    child = sn_make_core_child(par1 = hunter, par2 = hunter, scale = scale)
    logger.info(f"!_!! Родитель: {hunter.id}  toWins={toWins}     Потомок: {child.id}")

    if (margin := sn_eval_organism(child, toWins=toWins)) is None:
        return None
    if check_margin and margin[0] < 0:
        logger.info(f"!!_! Родитель: {hunter.id}  toWins={toWins}     Потомок: {child.id}   Проблема: 02 check_margin and margin[0] < 0")
        return None
    logger.info(f"!!_! Родитель: {hunter.id}  toWins={toWins}     Потомок: {child.id}   ОК")
    return child


def sn_make_complex_child(hunter: str):
    genotype = sn_get_org_by_id(hunter).genotype
    sn_make_child(sn_get_org_by_id(hunter), toWins=150, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=140, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=130, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=120, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=110, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=100, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=90, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=80, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=70, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=60, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=50, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=40, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=30, scale=0, check_margin=False)
    sn_make_child(sn_get_org_by_id(hunter), toWins=20, scale=0, check_margin=False)
    sn_get_org_by_id(hunter).die()






genotype = Genotype()

_tickers = None
#_end = None


#step = 1
#org = None
#_setup()


#step = _step_setup(step)   ############:
d_min, d_max = population.min_max_date()
if _tickers is None:
    _tickers = load_tickers()
#    _end = d_max or listing.all_history_date(_tickers)[-1]
    _end = d_max or all_history_date(_tickers)[-1]

#dates = listing.all_history_date(_tickers, start=_end)
dates = all_history_date(_tickers, start=_end)
#if (d_min != _end) or (len(dates) == 1):
#    step = step + 0
##    print(step + 1)


import logging
logger = logging.getLogger()
collection = store.get_collection()

"""
sn_make_complex_child("635189748293e7ae3519118f")	#2022-10-23  00:10		предполагаю 150 часов = 6,25 - 7 дней.
sn_make_complex_child("635189748293e7ae35191190")
sn_make_complex_child("635189748293e7ae35191191")
sn_make_complex_child("635189748293e7ae35191192")
sn_make_complex_child("635189748293e7ae35191193")
sn_make_complex_child("635189748293e7ae35191194")
sn_make_complex_child("635189748293e7ae35191195")
sn_make_complex_child("635189748293e7ae35191196")
sn_make_complex_child("635189748293e7ae35191197")
sn_make_complex_child("635189748293e7ae35191198")
sn_make_complex_child("635189748293e7ae35191199")
sn_make_complex_child("635189748293e7ae3519119a")
sn_make_complex_child("635189748293e7ae3519119b")
sn_make_complex_child("635189748293e7ae3519119c")
sn_make_complex_child("635189748293e7ae3519119d")
sn_make_complex_child("635189748293e7ae3519119e")
sn_make_complex_child("635189748293e7ae3519119f")
sn_make_complex_child("635189748293e7ae351911a0")
sn_make_complex_child("635189748293e7ae351911a1")
sn_make_complex_child("635189748293e7ae351911a2")
sn_make_complex_child("635189748293e7ae351911a3")	#2020-10-23 00:14   8 real
sn_make_complex_child("635189748293e7ae351911a4")
sn_make_complex_child("635189748293e7ae351911a5")
sn_make_complex_child("635189748293e7ae351911a6")
sn_make_complex_child("635189748293e7ae351911a7")
sn_make_complex_child("635189748293e7ae351911a8")
sn_make_complex_child("635189748293e7ae351911a9")
sn_make_complex_child("635189748293e7ae351911aa")
sn_make_complex_child("635189748293e7ae351911ab")
sn_make_complex_child("635189748293e7ae351911ac")
sn_make_complex_child("635189748293e7ae351911ad")
sn_make_complex_child("635189748293e7ae351911ae")
sn_make_complex_child("635189748293e7ae351911af")
sn_make_complex_child("635189748293e7ae351911b0")
sn_make_complex_child("635189748293e7ae351911b1")
sn_make_complex_child("635189748293e7ae351911b2")
sn_make_complex_child("635189748293e7ae351911b3")
sn_make_complex_child("635189748293e7ae351911b4")
sn_make_complex_child("635189748293e7ae351911b5")
sn_make_complex_child("635189748293e7ae351911b6")
sn_make_complex_child("635189748293e7ae351911b7")
sn_make_complex_child("635189748293e7ae351911b8")
sn_make_complex_child("635189748293e7ae351911b9")
sn_make_complex_child("635189748293e7ae351911ba")
sn_make_complex_child("635189748293e7ae351911bb")
sn_make_complex_child("635189748293e7ae351911bc")
sn_make_complex_child("635189748293e7ae351911bd")
sn_make_complex_child("635189748293e7ae351911be")
sn_make_complex_child("635189748293e7ae351911bf")
sn_make_complex_child("635189748293e7ae351911c0")
sn_make_complex_child("635189748293e7ae351911c1")
sn_make_complex_child("635189748293e7ae351911c2")
sn_make_complex_child("635189748293e7ae351911c3")
sn_make_complex_child("635189748293e7ae351911c4")
sn_make_complex_child("635189748293e7ae351911c5")
sn_make_complex_child("635189748293e7ae351911c6")
sn_make_complex_child("635189748293e7ae351911c7")
sn_make_complex_child("635189748293e7ae351911c8")
sn_make_complex_child("635189748293e7ae351911c9")
sn_make_complex_child("635189748293e7ae351911ca")
sn_make_complex_child("635189748293e7ae351911cb")
sn_make_complex_child("635189748293e7ae351911cc")
sn_make_complex_child("635189748293e7ae351911cd")
sn_make_complex_child("635189748293e7ae351911ce")
sn_make_complex_child("635189748293e7ae351911cf")
sn_make_complex_child("635189748293e7ae351911d0")
sn_make_complex_child("635189748293e7ae351911d1")
sn_make_complex_child("635189748293e7ae351911d2")
sn_make_complex_child("635189748293e7ae351911d3")
sn_make_complex_child("635189748293e7ae351911d4")
sn_make_complex_child("635189748293e7ae351911d5")
sn_make_complex_child("635189748293e7ae351911d6")
sn_make_complex_child("635189748293e7ae351911d7")
sn_make_complex_child("635189748293e7ae351911d8")
sn_make_complex_child("635189748293e7ae351911d9")
sn_make_complex_child("635189748293e7ae351911da")
sn_make_complex_child("635189748293e7ae351911db")
sn_make_complex_child("635189748293e7ae351911dc")
sn_make_complex_child("635189748293e7ae351911dd")
sn_make_complex_child("635189748293e7ae351911de")
sn_make_complex_child("635189748293e7ae351911df")
sn_make_complex_child("635189748293e7ae351911e0")
sn_make_complex_child("635189748293e7ae351911e1")
sn_make_complex_child("635189748293e7ae351911e2")
sn_make_complex_child("635189748293e7ae351911e3")
sn_make_complex_child("635189748293e7ae351911e4")
sn_make_complex_child("635189748293e7ae351911e5")
sn_make_complex_child("635189748293e7ae351911e6")
sn_make_complex_child("635189748293e7ae351911e7")
sn_make_complex_child("635189748293e7ae351911e8")
sn_make_complex_child("635189748293e7ae351911e9")
sn_make_complex_child("635189748293e7ae351911ea")
sn_make_complex_child("635189748293e7ae351911eb")
sn_make_complex_child("635189748293e7ae351911ec")
sn_make_complex_child("635189748293e7ae351911ed")
sn_make_complex_child("635189748293e7ae351911ee")
sn_make_complex_child("635189748293e7ae351911ef")
sn_make_complex_child("635189748293e7ae351911f0")
sn_make_complex_child("635189748293e7ae351911f1")
sn_make_complex_child("635189748293e7ae351911f2")
sn_make_complex_child("635189748293e7ae351911f3")
sn_make_complex_child("635189748293e7ae351911f4")
sn_make_complex_child("635189748293e7ae351911f5")
sn_make_complex_child("635189748293e7ae351911f6")
sn_make_complex_child("635189748293e7ae351911f7")
sn_make_complex_child("635189748293e7ae351911f8")
sn_make_complex_child("635189748293e7ae351911f9")
sn_make_complex_child("635189748293e7ae351911fa")
sn_make_complex_child("635189748293e7ae351911fb")
sn_make_complex_child("635189748293e7ae351911fc")
sn_make_complex_child("635189748293e7ae351911fd")
sn_make_complex_child("635189748293e7ae351911fe")
sn_make_complex_child("635189748293e7ae351911ff")
sn_make_complex_child("635189748293e7ae35191200")
sn_make_complex_child("635189748293e7ae35191201")
sn_make_complex_child("635189748293e7ae35191202")
sn_make_complex_child("635189748293e7ae35191203")
sn_make_complex_child("635189748293e7ae35191204")
sn_make_complex_child("635189748293e7ae35191205")
sn_make_complex_child("635189748293e7ae35191206")
sn_make_complex_child("635189748293e7ae35191207")
sn_make_complex_child("635189748293e7ae35191208")
sn_make_complex_child("635189748293e7ae35191209")
sn_make_complex_child("635189748293e7ae3519120a")
sn_make_complex_child("635189748293e7ae3519120b")
sn_make_complex_child("635189748293e7ae3519120c")
sn_make_complex_child("635189748293e7ae3519120d")
sn_make_complex_child("635189748293e7ae3519120e")
sn_make_complex_child("635189748293e7ae3519120f")
sn_make_complex_child("635189748293e7ae35191210")
sn_make_complex_child("635189748293e7ae35191211")
sn_make_complex_child("635189748293e7ae35191212")
sn_make_complex_child("635189748293e7ae35191213")
sn_make_complex_child("635189748293e7ae35191214")
sn_make_complex_child("635189748293e7ae35191215")
sn_make_complex_child("635189748293e7ae35191216")
sn_make_complex_child("635189748293e7ae35191217")
sn_make_complex_child("635189748293e7ae35191218")
sn_make_complex_child("635189748293e7ae35191219")
sn_make_complex_child("635189748293e7ae3519121a")
sn_make_complex_child("635189748293e7ae3519121b")
sn_make_complex_child("635189748293e7ae3519121c")
sn_make_complex_child("635189748293e7ae3519121d")
sn_make_complex_child("635189748293e7ae3519121e")
sn_make_complex_child("635189748293e7ae3519121f")
sn_make_complex_child("635189748293e7ae35191220")
sn_make_complex_child("635189748293e7ae35191221")
sn_make_complex_child("635189748293e7ae35191222")
sn_make_complex_child("635189748293e7ae35191223")
sn_make_complex_child("635189748293e7ae35191224")
sn_make_complex_child("635189748293e7ae35191225")
sn_make_complex_child("635189748293e7ae35191226")
sn_make_complex_child("635189748293e7ae35191227")
sn_make_complex_child("635189748293e7ae35191228")
sn_make_complex_child("635189748293e7ae35191229")
sn_make_complex_child("635189748293e7ae3519122a")
sn_make_complex_child("635189748293e7ae3519122b")
sn_make_complex_child("635189748293e7ae3519122c")
sn_make_complex_child("635189748293e7ae3519122d")
sn_make_complex_child("635189748293e7ae3519122e")
sn_make_complex_child("635189748293e7ae3519122f")
sn_make_complex_child("635189748293e7ae35191230")
sn_make_complex_child("635189748293e7ae35191231")
sn_make_complex_child("635189748293e7ae35191232")
sn_make_complex_child("635189748293e7ae35191233")
sn_make_complex_child("635189748293e7ae35191234")
sn_make_complex_child("635189748293e7ae35191235")
sn_make_complex_child("635189748293e7ae35191236")
sn_make_complex_child("635189748293e7ae35191237")
sn_make_complex_child("635189748293e7ae35191238")
sn_make_complex_child("635189748293e7ae35191239")
sn_make_complex_child("635189748293e7ae3519123a")
sn_make_complex_child("635189748293e7ae3519123b")
sn_make_complex_child("635189748293e7ae3519123c")
sn_make_complex_child("635189748293e7ae3519123d")
sn_make_complex_child("635189748293e7ae3519123e")
sn_make_complex_child("635189748293e7ae3519123f")
sn_make_complex_child("635189748293e7ae35191240")
sn_make_complex_child("635189748293e7ae35191241")
sn_make_complex_child("635189748293e7ae35191242")
sn_make_complex_child("635189748293e7ae35191243")
sn_make_complex_child("635189748293e7ae35191244")
sn_make_complex_child("635189748293e7ae35191245")
sn_make_complex_child("635189748293e7ae35191246")
sn_make_complex_child("635189748293e7ae35191247")
sn_make_complex_child("635189748293e7ae35191248")
sn_make_complex_child("635189748293e7ae35191249")
sn_make_complex_child("635189748293e7ae3519124a")
sn_make_complex_child("635189748293e7ae3519124b")
sn_make_complex_child("635189748293e7ae3519124c")
sn_make_complex_child("635189748293e7ae3519124d")
sn_make_complex_child("635189748293e7ae3519124e")
sn_make_complex_child("635189748293e7ae3519124f")
sn_make_complex_child("635189748293e7ae35191250")
sn_make_complex_child("635189748293e7ae35191251")
sn_make_complex_child("635189748293e7ae35191252")
sn_make_complex_child("635189748293e7ae35191253")
sn_make_complex_child("635189748293e7ae35191254")
sn_make_complex_child("635189748293e7ae35191255")
sn_make_complex_child("635189748293e7ae35191256")
sn_make_complex_child("635189748293e7ae35191257")
sn_make_complex_child("635189748293e7ae35191258")
sn_make_complex_child("635189748293e7ae35191259")
sn_make_complex_child("635189748293e7ae3519125a")
sn_make_complex_child("635189748293e7ae3519125b")
sn_make_complex_child("635189748293e7ae3519125c")
sn_make_complex_child("635189748293e7ae3519125d")
sn_make_complex_child("635189748293e7ae3519125e")
sn_make_complex_child("635189748293e7ae3519125f")
sn_make_complex_child("635189748293e7ae35191260")
sn_make_complex_child("635189748293e7ae35191261")
sn_make_complex_child("635189748293e7ae35191262")
sn_make_complex_child("635189748293e7ae35191263")
sn_make_complex_child("635189748293e7ae35191264")
sn_make_complex_child("635189748293e7ae35191265")
sn_make_complex_child("635189748293e7ae35191266")
sn_make_complex_child("635189748293e7ae35191267")
sn_make_complex_child("635189748293e7ae35191268")
sn_make_complex_child("635189748293e7ae35191269")
sn_make_complex_child("635189748293e7ae3519126a")
sn_make_complex_child("635189748293e7ae3519126b")
sn_make_complex_child("635189748293e7ae3519126c")
sn_make_complex_child("635189748293e7ae3519126d")
sn_make_complex_child("635189748293e7ae3519126e")
sn_make_complex_child("635189748293e7ae3519126f")
sn_make_complex_child("635189748293e7ae35191270")
sn_make_complex_child("635189748293e7ae35191271")
sn_make_complex_child("635189748293e7ae35191272")
sn_make_complex_child("635189748293e7ae35191273")
sn_make_complex_child("635189748293e7ae35191274")
sn_make_complex_child("635189748293e7ae35191275")
sn_make_complex_child("635189748293e7ae35191276")
sn_make_complex_child("635189748293e7ae35191277")
sn_make_complex_child("635189748293e7ae35191278")
sn_make_complex_child("635189748293e7ae35191279")
sn_make_complex_child("635189748293e7ae3519127a")
sn_make_complex_child("635189748293e7ae3519127b")
sn_make_complex_child("635189748293e7ae3519127c")
sn_make_complex_child("635189748293e7ae3519127d")
sn_make_complex_child("635189748293e7ae3519127e")
sn_make_complex_child("635189748293e7ae3519127f")
sn_make_complex_child("635189748293e7ae35191280")
sn_make_complex_child("635189748293e7ae35191281")
sn_make_complex_child("635189748293e7ae35191282")
sn_make_complex_child("635189748293e7ae35191283")
sn_make_complex_child("635189748293e7ae35191284")
sn_make_complex_child("635189748293e7ae35191285")
sn_make_complex_child("635189748293e7ae35191286")
sn_make_complex_child("635189748293e7ae35191287")
sn_make_complex_child("635189748293e7ae35191288")
sn_make_complex_child("635189748293e7ae35191289")
sn_make_complex_child("635189748293e7ae3519128a")
sn_make_complex_child("635189748293e7ae3519128b")
sn_make_complex_child("635189748293e7ae3519128c")
sn_make_complex_child("635189748293e7ae3519128d")
sn_make_complex_child("635189748293e7ae3519128e")
sn_make_complex_child("635189748293e7ae3519128f")
sn_make_complex_child("635189748293e7ae35191290")
sn_make_complex_child("635189748293e7ae35191291")
sn_make_complex_child("635189748293e7ae35191292")
sn_make_complex_child("635189748293e7ae35191293")
sn_make_complex_child("635189748293e7ae35191294")
sn_make_complex_child("635189748293e7ae35191295")
sn_make_complex_child("635189748293e7ae35191296")
sn_make_complex_child("635189748293e7ae35191297")
sn_make_complex_child("635189748293e7ae35191298")
sn_make_complex_child("635189748293e7ae35191299")
sn_make_complex_child("635189748293e7ae3519129a")
sn_make_complex_child("635189748293e7ae3519129b")
sn_make_complex_child("635189748293e7ae3519129c")
sn_make_complex_child("635189748293e7ae3519129d")
sn_make_complex_child("635189748293e7ae3519129e")
sn_make_complex_child("635189748293e7ae3519129f")
sn_make_complex_child("635189748293e7ae351912a0")
sn_make_complex_child("635189748293e7ae351912a1")
sn_make_complex_child("635189748293e7ae351912a2")
sn_make_complex_child("635189748293e7ae351912a3")
sn_make_complex_child("635189748293e7ae351912a4")
sn_make_complex_child("635189748293e7ae351912a5")
sn_make_complex_child("635189748293e7ae351912a6")
sn_make_complex_child("635189748293e7ae351912a7")
sn_make_complex_child("635189748293e7ae351912a8")
sn_make_complex_child("635189748293e7ae351912a9")
"""
sn_make_complex_child("635189748293e7ae351912aa")
sn_make_complex_child("635189748293e7ae351912ab")
sn_make_complex_child("635189748293e7ae351912ac")
sn_make_complex_child("635189748293e7ae351912ad")
sn_make_complex_child("635189748293e7ae351912ae")
sn_make_complex_child("635189748293e7ae351912af")
sn_make_complex_child("635189748293e7ae351912b0")
sn_make_complex_child("635189748293e7ae351912b1")
sn_make_complex_child("635189748293e7ae351912b2")
sn_make_complex_child("635189748293e7ae351912b3")
sn_make_complex_child("635189748293e7ae351912b4")
sn_make_complex_child("635189748293e7ae351912b5")
#sn_make_complex_child("635189748293e7ae351912b6")



quit()


















#sn_step_parent(sn_get_org_by_id("633b087ee8249177d2e3eae5"), to_scores=112, childs=0, number=1)
#sn_step_parent(sn_get_org_by_id("633ef9b0d88bc0499b8426c1"), to_scores=111, childs=0, number=2)
#sn_step_parent(sn_get_org_by_id("6338c3a5e8249177d2e3eaa4"), to_scores=110, childs=0, number=3)
#sn_step_parent(sn_get_org_by_id("6338c18be8249177d2e3eaa3"), to_scores=109, childs=0, number=4)
#sn_step_parent(sn_get_org_by_id("633eeb22d88bc0499b8426be"), to_scores=109, childs=0, number=5)
#sn_step_parent(sn_get_org_by_id("633712504c9e716ac9ea156b"), to_scores=107, childs=0, number=6)
#sn_step_parent(sn_get_org_by_id("633ee1f1d88bc0499b8426bc"), to_scores=107, childs=0, number=7)
#sn_step_parent(sn_get_org_by_id("633fad3983329d94dcefdee3"), to_scores=107, childs=0, number=8)
#sn_step_parent(sn_get_org_by_id("6338bd05e8249177d2e3eaa1"), to_scores=105, childs=0, number=9)
#sn_step_parent(sn_get_org_by_id("633d79156ea1d94a4552b863"), to_scores=105, childs=0, number=10)
#sn_step_parent(sn_get_org_by_id("633633254c9e716ac9ea1551"), to_scores=104, childs=0, number=11)
#sn_step_parent(sn_get_org_by_id("6331fe79264cd29eee372fa0"), to_scores=103, childs=0, number=12)
#sn_step_parent(sn_get_org_by_id("63362f574c9e716ac9ea1550"), to_scores=103, childs=0, number=13)
#sn_step_parent(sn_get_org_by_id("6335e7d9febdd9e42310fba3"), to_scores=103, childs=0, number=14)
#sn_step_parent(sn_get_org_by_id("63383d2453ee249acb5e5c63"), to_scores=103, childs=0, number=15)
#sn_step_parent(sn_get_org_by_id("633834be53ee249acb5e5c61"), to_scores=102, childs=0, number=16)
#sn_step_parent(sn_get_org_by_id("63316809ec4c52145d986b8d"), to_scores=101, childs=0, number=17)
#sn_step_parent(sn_get_org_by_id("633c36eae8249177d2e3eb0c"), to_scores=101, childs=0, number=18)
#sn_step_parent(sn_get_org_by_id("633e0a13d88bc0499b84269f"), to_scores=101, childs=0, number=19)
#sn_step_parent(sn_get_org_by_id("633cde2a6ea1d94a4552b84a"), to_scores=96, childs=0, number=20)
#sn_step_parent(sn_get_org_by_id("6333e6c8c60722cd4926e973"), to_scores=91, childs=0, number=21)
#sn_step_parent(sn_get_org_by_id("6350d437e980e7cc1b758dfb"), to_scores=86, childs=0, number=22)
#sn_step_parent(sn_get_org_by_id("6350cd1ae980e7cc1b758df8"), to_scores=85, childs=0, number=23)
#sn_step_parent(sn_get_org_by_id("6350c2454d2d42658f0f611d"), to_scores=84, childs=0, number=24)
#sn_step_parent(sn_get_org_by_id("6350bae84d2d42658f0f6116"), to_scores=83, childs=0, number=25)
#sn_step_parent(sn_get_org_by_id("6350a2bc4d2d42658f0f610e"), to_scores=82, childs=0, number=26)
#sn_step_parent(sn_get_org_by_id("635089e04d2d42658f0f6101"), to_scores=81, childs=0, number=27)
#sn_step_parent(sn_get_org_by_id("63509de34d2d42658f0f610a"), to_scores=81, childs=0, number=28)
#sn_step_parent(sn_get_org_by_id("6350886f4d2d42658f0f6100"), to_scores=80, childs=0, number=29)
#sn_step_parent(sn_get_org_by_id("63508d654d2d42658f0f6104"), to_scores=80, childs=0, number=30)
#sn_step_parent(sn_get_org_by_id("635086ff4d2d42658f0f60ff"), to_scores=79, childs=0, number=31)
#sn_step_parent(sn_get_org_by_id("635085ab4d2d42658f0f60fe"), to_scores=78, childs=0, number=32)
#sn_step_parent(sn_get_org_by_id("635082494d2d42658f0f60fc"), to_scores=77, childs=0, number=33)
#sn_step_parent(sn_get_org_by_id("63507cc34d2d42658f0f60f8"), to_scores=76, childs=0, number=34)
#sn_step_parent(sn_get_org_by_id("635080a84d2d42658f0f60fb"), to_scores=76, childs=0, number=35)
#sn_step_parent(sn_get_org_by_id("63507b674d2d42658f0f60f7"), to_scores=75, childs=0, number=36)
#sn_step_parent(sn_get_org_by_id("63507dd04d2d42658f0f60f9"), to_scores=75, childs=0, number=37)
#sn_step_parent(sn_get_org_by_id("635078884d2d42658f0f60f5"), to_scores=74, childs=0, number=38)
#sn_step_parent(sn_get_org_by_id("6350773f4d2d42658f0f60f4"), to_scores=73, childs=0, number=39)
#sn_step_parent(sn_get_org_by_id("63506f8f81dc179c95b0c3dd"), to_scores=72, childs=0, number=40)
#sn_step_parent(sn_get_org_by_id("6350689e81dc179c95b0c3d9"), to_scores=71, childs=0, number=41)
#sn_step_parent(sn_get_org_by_id("63506dbe81dc179c95b0c3dc"), to_scores=71, childs=0, number=42)
#sn_step_parent(sn_get_org_by_id("6350673e81dc179c95b0c3d8"), to_scores=70, childs=0, number=43)
#sn_step_parent(sn_get_org_by_id("63506be981dc179c95b0c3db"), to_scores=70, childs=0, number=44)
#sn_step_parent(sn_get_org_by_id("63505b567f2e7b05a07001d2"), to_scores=69, childs=0, number=45)
#sn_step_parent(sn_get_org_by_id("63504ba57f2e7b05a07001c3"), to_scores=68, childs=0, number=46)
#sn_step_parent(sn_get_org_by_id("632ca3f21bd8434c1e6134ba"), to_scores=67, childs=0, number=47)
#sn_step_parent(sn_get_org_by_id("632d3d6c1bd8434c1e6134db"), to_scores=67, childs=0, number=48)
#sn_step_parent(sn_get_org_by_id("635040ce7f2e7b05a07001c0"), to_scores=67, childs=0, number=49)
#sn_step_parent(sn_get_org_by_id("632ad2429dbf8ba893b5e25a"), to_scores=66, childs=0, number=50)
#sn_step_parent(sn_get_org_by_id("6345d2ff636053dd92556e00"), to_scores=66, childs=0, number=51)
#sn_step_parent(sn_get_org_by_id("635031c5e482468269dd5965"), to_scores=66, childs=0, number=52)
#sn_step_parent(sn_get_org_by_id("63503f327f2e7b05a07001bf"), to_scores=66, childs=0, number=53)
#sn_step_parent(sn_get_org_by_id("632c1e0ff301bb5b84ca7af6"), to_scores=65, childs=0, number=54)
#sn_step_parent(sn_get_org_by_id("632c81d5e7daf3732b117ded"), to_scores=65, childs=0, number=55)
#sn_step_parent(sn_get_org_by_id("635022d8e482468269dd5956"), to_scores=65, childs=0, number=56)
#sn_step_parent(sn_get_org_by_id("6350346ce482468269dd5967"), to_scores=65, childs=0, number=57)
#sn_step_parent(sn_get_org_by_id("6345c091636053dd92556df8"), to_scores=64, childs=0, number=58)
#sn_step_parent(sn_get_org_by_id("63502161e482468269dd5955"), to_scores=64, childs=0, number=59)
#sn_step_parent(sn_get_org_by_id("63501a89e482468269dd5951"), to_scores=63, childs=0, number=60)
#sn_step_parent(sn_get_org_by_id("63458eeb636053dd92556de8"), to_scores=62, childs=0, number=61)
#sn_step_parent(sn_get_org_by_id("6350180ae482468269dd5950"), to_scores=62, childs=0, number=62)
#sn_step_parent(sn_get_org_by_id("63500d27e482468269dd594b"), to_scores=61, childs=0, number=63)
#sn_step_parent(sn_get_org_by_id("635015d9e482468269dd594f"), to_scores=61, childs=0, number=64)
#sn_step_parent(sn_get_org_by_id("63456af8636053dd92556dd7"), to_scores=60, childs=0, number=65)
#sn_step_parent(sn_get_org_by_id("63457e10636053dd92556de0"), to_scores=60, childs=0, number=66)
#sn_step_parent(sn_get_org_by_id("63500896e482468269dd5949"), to_scores=60, childs=0, number=67)
#sn_step_parent(sn_get_org_by_id("635011f1e482468269dd594d"), to_scores=60, childs=0, number=68)
#sn_step_parent(sn_get_org_by_id("634568eb636053dd92556dd5"), to_scores=59, childs=0, number=69)
#sn_step_parent(sn_get_org_by_id("63500670e482468269dd5948"), to_scores=59, childs=0, number=70)
#sn_step_parent(sn_get_org_by_id("63500435e482468269dd5947"), to_scores=58, childs=0, number=71)
#sn_step_parent(sn_get_org_by_id("634556ed636053dd92556dca"), to_scores=57, childs=0, number=72)
#sn_step_parent(sn_get_org_by_id("634fffa0e482468269dd5945"), to_scores=57, childs=0, number=73)
#sn_step_parent(sn_get_org_by_id("634549b5636053dd92556dc4"), to_scores=56, childs=0, number=74)
#sn_step_parent(sn_get_org_by_id("6349bd64a5657a22c93bcb63"), to_scores=56, childs=0, number=75)
#sn_step_parent(sn_get_org_by_id("634ff783e482468269dd5940"), to_scores=56, childs=0, number=76)
#sn_step_parent(sn_get_org_by_id("634fc79bb7d58c53b1d8ae65"), to_scores=55, childs=0, number=77)
#sn_step_parent(sn_get_org_by_id("634ffa77e482468269dd5942"), to_scores=55, childs=0, number=78)
#sn_step_parent(sn_get_org_by_id("634fc46fb7d58c53b1d8ae63"), to_scores=54, childs=0, number=79)
#sn_step_parent(sn_get_org_by_id("634fbe5bb7d58c53b1d8ae60"), to_scores=53, childs=0, number=80)
#sn_step_parent(sn_get_org_by_id("6349a280a5657a22c93bcb52"), to_scores=52, childs=0, number=81)
#sn_step_parent(sn_get_org_by_id("634fbc9ab7d58c53b1d8ae5f"), to_scores=52, childs=0, number=82)
#sn_step_parent(sn_get_org_by_id("634fb434b7d58c53b1d8ae5a"), to_scores=51, childs=0, number=83)
#sn_step_parent(sn_get_org_by_id("634fba27b7d58c53b1d8ae5d"), to_scores=51, childs=0, number=84)
#sn_step_parent(sn_get_org_by_id("6344f8e01a85833dfcdfde7c"), to_scores=50, childs=0, number=85)
#sn_step_parent(sn_get_org_by_id("6349817ba5657a22c93bcb3b"), to_scores=50, childs=0, number=86)
#sn_step_parent(sn_get_org_by_id("634f1810ea63d87f1925722f"), to_scores=50, childs=0, number=87)
#sn_step_parent(sn_get_org_by_id("634fac9bb7d58c53b1d8ae55"), to_scores=50, childs=0, number=88)
#sn_step_parent(sn_get_org_by_id("634fb5f3b7d58c53b1d8ae5b"), to_scores=50, childs=0, number=89)
#sn_step_parent(sn_get_org_by_id("6344f7751a85833dfcdfde7b"), to_scores=49, childs=0, number=90)
#sn_step_parent(sn_get_org_by_id("634f13cdea63d87f1925722e"), to_scores=49, childs=0, number=91)
#sn_step_parent(sn_get_org_by_id("634fa144b7d58c53b1d8ae52"), to_scores=49, childs=0, number=92)
#sn_step_parent(sn_get_org_by_id("634f0fd3ea63d87f1925722d"), to_scores=48, childs=0, number=93)
#sn_step_parent(sn_get_org_by_id("634f97a7b7d58c53b1d8ae4e"), to_scores=48, childs=0, number=94)
#sn_step_parent(sn_get_org_by_id("634f0de2ea63d87f1925722c"), to_scores=47, childs=0, number=95)
#sn_step_parent(sn_get_org_by_id("634f91c8b7d58c53b1d8ae4b"), to_scores=47, childs=0, number=96)
#sn_step_parent(sn_get_org_by_id("6344eb8c1a85833dfcdfde73"), to_scores=46, childs=0, number=97)
#sn_step_parent(sn_get_org_by_id("6344da931a85833dfcdfde6e"), to_scores=46, childs=0, number=98)
#sn_step_parent(sn_get_org_by_id("634954e2a5657a22c93bcb1f"), to_scores=46, childs=0, number=99)
#sn_step_parent(sn_get_org_by_id("634ef7afea63d87f19257222"), to_scores=46, childs=0, number=100)
#sn_step_parent(sn_get_org_by_id("634f03a1ea63d87f19257227"), to_scores=46, childs=0, number=101)
#sn_step_parent(sn_get_org_by_id("634f902ab7d58c53b1d8ae4a"), to_scores=46, childs=0, number=102)
#sn_step_parent(sn_get_org_by_id("6344d8e31a85833dfcdfde6d"), to_scores=45, childs=0, number=103)
#sn_step_parent(sn_get_org_by_id("634efe95ea63d87f19257224"), to_scores=45, childs=0, number=104)
#sn_step_parent(sn_get_org_by_id("6344a55b1a85833dfcdfde4b"), to_scores=44, childs=0, number=105)
#sn_step_parent(sn_get_org_by_id("6344a4741a85833dfcdfde4a"), to_scores=44, childs=0, number=106)
#sn_step_parent(sn_get_org_by_id("6344a9291a85833dfcdfde4f"), to_scores=44, childs=0, number=107)
#sn_step_parent(sn_get_org_by_id("6344ac8c1a85833dfcdfde52"), to_scores=44, childs=0, number=108)
#sn_step_parent(sn_get_org_by_id("6344b0291a85833dfcdfde55"), to_scores=44, childs=0, number=109)
#sn_step_parent(sn_get_org_by_id("6344c4181a85833dfcdfde66"), to_scores=44, childs=0, number=110)
#sn_step_parent(sn_get_org_by_id("634eeeb9ea63d87f1925721e"), to_scores=44, childs=0, number=111)
#sn_step_parent(sn_get_org_by_id("6344b3ee1a85833dfcdfde58"), to_scores=43, childs=0, number=112)
#sn_step_parent(sn_get_org_by_id("6344bc571a85833dfcdfde5f"), to_scores=43, childs=0, number=113)
#sn_step_parent(sn_get_org_by_id("634936f2ac2afce6962bf329"), to_scores=43, childs=0, number=114)
#sn_step_parent(sn_get_org_by_id("634ee7c3ea63d87f1925721a"), to_scores=43, childs=0, number=115)
#sn_step_parent(sn_get_org_by_id("634d67d25f8bfcffdaddc4a2"), to_scores=42, childs=0, number=116)
#sn_step_parent(sn_get_org_by_id("634d70ab5f8bfcffdaddc4a8"), to_scores=42, childs=0, number=117)
#sn_step_parent(sn_get_org_by_id("634e8d1c1d4ff37def55660c"), to_scores=42, childs=0, number=118)
#sn_step_parent(sn_get_org_by_id("634934c1ac2afce6962bf327"), to_scores=41, childs=0, number=119)
#sn_step_parent(sn_get_org_by_id("63493341ac2afce6962bf325"), to_scores=41, childs=0, number=120)
#sn_step_parent(sn_get_org_by_id("634c9b0c5d9f8d87b1b02e3d"), to_scores=41, childs=0, number=121)
#sn_step_parent(sn_get_org_by_id("634ca0135d9f8d87b1b02e40"), to_scores=41, childs=0, number=122)
#sn_step_parent(sn_get_org_by_id("634d60965f8bfcffdaddc49c"), to_scores=41, childs=0, number=123)
#sn_step_parent(sn_get_org_by_id("634d64765f8bfcffdaddc49f"), to_scores=41, childs=0, number=124)
#sn_step_parent(sn_get_org_by_id("634e58201d4ff37def5565ea"), to_scores=41, childs=0, number=125)
#sn_step_parent(sn_get_org_by_id("634e885c1d4ff37def556609"), to_scores=41, childs=0, number=126)
#sn_step_parent(sn_get_org_by_id("634e8b7b1d4ff37def55660b"), to_scores=41, childs=0, number=127)
#sn_step_parent(sn_get_org_by_id("634ccf7e43694618d07b77a9"), to_scores=40, childs=0, number=128)
#sn_step_parent(sn_get_org_by_id("634cd36f43694618d07b77ab"), to_scores=40, childs=0, number=129)
#sn_step_parent(sn_get_org_by_id("634d25095f8bfcffdaddc47c"), to_scores=40, childs=0, number=130)
#sn_step_parent(sn_get_org_by_id("634d5f1d5f8bfcffdaddc49b"), to_scores=40, childs=0, number=131)
#sn_step_parent(sn_get_org_by_id("634e89ff1d4ff37def55660a"), to_scores=40, childs=0, number=132)
#sn_step_parent(sn_get_org_by_id("63491ee2ac2afce6962bf319"), to_scores=39, childs=0, number=133)
#sn_step_parent(sn_get_org_by_id("634b7b761c108ceacebb2f2f"), to_scores=39, childs=0, number=134)
#sn_step_parent(sn_get_org_by_id("634cb1cd5d9f8d87b1b02e4a"), to_scores=39, childs=0, number=135)
#sn_step_parent(sn_get_org_by_id("634d23da5f8bfcffdaddc47b"), to_scores=39, childs=0, number=136)
#sn_step_parent(sn_get_org_by_id("634d5dd15f8bfcffdaddc49a"), to_scores=39, childs=0, number=137)
#sn_step_parent(sn_get_org_by_id("634e7cc11d4ff37def556601"), to_scores=39, childs=0, number=138)
#sn_step_parent(sn_get_org_by_id("634c646d9b83cea15a2b9eb9"), to_scores=38, childs=0, number=139)
#sn_step_parent(sn_get_org_by_id("634ccb5c43694618d07b77a6"), to_scores=38, childs=0, number=140)
#sn_step_parent(sn_get_org_by_id("634d0c645f8bfcffdaddc46f"), to_scores=38, childs=0, number=141)
#sn_step_parent(sn_get_org_by_id("634d1ba65f8bfcffdaddc474"), to_scores=38, childs=0, number=142)
#sn_step_parent(sn_get_org_by_id("634d22ae5f8bfcffdaddc47a"), to_scores=38, childs=0, number=143)
#sn_step_parent(sn_get_org_by_id("634d5a075f8bfcffdaddc498"), to_scores=38, childs=0, number=144)
#sn_step_parent(sn_get_org_by_id("634e6dc71d4ff37def5565fb"), to_scores=38, childs=0, number=145)
#sn_step_parent(sn_get_org_by_id("63491d6eac2afce6962bf317"), to_scores=37, childs=0, number=146)
#sn_step_parent(sn_get_org_by_id("634c5fe39b83cea15a2b9eb6"), to_scores=37, childs=0, number=147)
#sn_step_parent(sn_get_org_by_id("634c631a9b83cea15a2b9eb8"), to_scores=37, childs=0, number=148)
#sn_step_parent(sn_get_org_by_id("634d21595f8bfcffdaddc479"), to_scores=37, childs=0, number=149)
#sn_step_parent(sn_get_org_by_id("634d2c685f8bfcffdaddc481"), to_scores=37, childs=0, number=150)
#sn_step_parent(sn_get_org_by_id("634d588d5f8bfcffdaddc497"), to_scores=37, childs=0, number=151)
#sn_step_parent(sn_get_org_by_id("634e6c241d4ff37def5565f9"), to_scores=37, childs=0, number=152)
#sn_step_parent(sn_get_org_by_id("63491cbfac2afce6962bf316"), to_scores=36, childs=0, number=153)
#sn_step_parent(sn_get_org_by_id("634b9b7195fd4b6edda478f3"), to_scores=36, childs=0, number=154)
#sn_step_parent(sn_get_org_by_id("634c5e8d9b83cea15a2b9eb5"), to_scores=36, childs=0, number=155)
#sn_step_parent(sn_get_org_by_id("634c70e75d9f8d87b1b02e1d"), to_scores=36, childs=0, number=156)
sn_step_parent(sn_get_org_by_id("634c6f715d9f8d87b1b02e1b"), to_scores=36, childs=0, number=157)
sn_step_parent(sn_get_org_by_id("634c79e65d9f8d87b1b02e25"), to_scores=36, childs=0, number=158)
sn_step_parent(sn_get_org_by_id("634cb44143694618d07b779b"), to_scores=36, childs=0, number=159)
sn_step_parent(sn_get_org_by_id("634cb56e43694618d07b779c"), to_scores=36, childs=0, number=160)
sn_step_parent(sn_get_org_by_id("634ce01bf9a895af0bdc8c66"), to_scores=36, childs=0, number=161)
sn_step_parent(sn_get_org_by_id("634cee14f9a895af0bdc8c6d"), to_scores=36, childs=0, number=162)
sn_step_parent(sn_get_org_by_id("634cf08bf9a895af0bdc8c70"), to_scores=36, childs=0, number=163)
sn_step_parent(sn_get_org_by_id("634cf2d9f9a895af0bdc8c72"), to_scores=36, childs=0, number=164)
sn_step_parent(sn_get_org_by_id("634cf3e6f9a895af0bdc8c73"), to_scores=36, childs=0, number=165)
sn_step_parent(sn_get_org_by_id("634cf81af9a895af0bdc8c76"), to_scores=36, childs=0, number=166)
sn_step_parent(sn_get_org_by_id("634cfa5bf9a895af0bdc8c78"), to_scores=36, childs=0, number=167)
sn_step_parent(sn_get_org_by_id("634d018f5f8bfcffdaddc465"), to_scores=36, childs=0, number=168)
sn_step_parent(sn_get_org_by_id("634d02b05f8bfcffdaddc466"), to_scores=36, childs=0, number=169)
sn_step_parent(sn_get_org_by_id("634d05b55f8bfcffdaddc468"), to_scores=36, childs=0, number=170)
sn_step_parent(sn_get_org_by_id("634d08cc5f8bfcffdaddc46b"), to_scores=36, childs=0, number=171)
sn_step_parent(sn_get_org_by_id("634d416b5f8bfcffdaddc48d"), to_scores=36, childs=0, number=172)
sn_step_parent(sn_get_org_by_id("634d40105f8bfcffdaddc48c"), to_scores=36, childs=0, number=173)
sn_step_parent(sn_get_org_by_id("634d42a75f8bfcffdaddc48e"), to_scores=36, childs=0, number=174)
sn_step_parent(sn_get_org_by_id("634d477c5f8bfcffdaddc490"), to_scores=36, childs=0, number=175)
sn_step_parent(sn_get_org_by_id("634d4c8e5f8bfcffdaddc493"), to_scores=36, childs=0, number=176)
sn_step_parent(sn_get_org_by_id("634d51185f8bfcffdaddc495"), to_scores=36, childs=0, number=177)
sn_step_parent(sn_get_org_by_id("634e49a71d4ff37def5565df"), to_scores=36, childs=0, number=178)
sn_step_parent(sn_get_org_by_id("634e52e31d4ff37def5565e5"), to_scores=36, childs=0, number=179)
sn_step_parent(sn_get_org_by_id("634e56631d4ff37def5565e8"), to_scores=36, childs=0, number=180)
sn_step_parent(sn_get_org_by_id("634e572e1d4ff37def5565e9"), to_scores=36, childs=0, number=181)
sn_step_parent(sn_get_org_by_id("634e58fb1d4ff37def5565eb"), to_scores=36, childs=0, number=182)
sn_step_parent(sn_get_org_by_id("634e59d21d4ff37def5565ec"), to_scores=36, childs=0, number=183)
sn_step_parent(sn_get_org_by_id("634e5abe1d4ff37def5565ed"), to_scores=36, childs=0, number=184)
sn_step_parent(sn_get_org_by_id("634e5d611d4ff37def5565ef"), to_scores=36, childs=0, number=185)
sn_step_parent(sn_get_org_by_id("634e5ecd1d4ff37def5565f0"), to_scores=36, childs=0, number=186)
sn_step_parent(sn_get_org_by_id("634e60a01d4ff37def5565f1"), to_scores=36, childs=0, number=187)
sn_step_parent(sn_get_org_by_id("634e63461d4ff37def5565f2"), to_scores=36, childs=0, number=188)
sn_step_parent(sn_get_org_by_id("634e6af01d4ff37def5565f8"), to_scores=36, childs=0, number=189)
sn_step_parent(sn_get_org_by_id("6346a9cbc98bdef76b6f8877"), to_scores=35, childs=0, number=190)
sn_step_parent(sn_get_org_by_id("63476a89ac2afce6962bf159"), to_scores=35, childs=0, number=191)
sn_step_parent(sn_get_org_by_id("634753b2ac2afce6962bf133"), to_scores=35, childs=0, number=192)
sn_step_parent(sn_get_org_by_id("63479b3aac2afce6962bf187"), to_scores=35, childs=0, number=193)
sn_step_parent(sn_get_org_by_id("6347bc44ac2afce6962bf1be"), to_scores=35, childs=0, number=194)
sn_step_parent(sn_get_org_by_id("63479b97ac2afce6962bf188"), to_scores=35, childs=0, number=195)
sn_step_parent(sn_get_org_by_id("6347df00ac2afce6962bf1da"), to_scores=35, childs=0, number=196)
sn_step_parent(sn_get_org_by_id("63486846ac2afce6962bf26f"), to_scores=35, childs=0, number=197)
sn_step_parent(sn_get_org_by_id("634818c0ac2afce6962bf22b"), to_scores=35, childs=0, number=198)
sn_step_parent(sn_get_org_by_id("634874e3ac2afce6962bf27c"), to_scores=35, childs=0, number=199)
sn_step_parent(sn_get_org_by_id("6348e32eac2afce6962bf2dc"), to_scores=35, childs=0, number=200)
sn_step_parent(sn_get_org_by_id("6348e07cac2afce6962bf2d8"), to_scores=35, childs=0, number=201)
sn_step_parent(sn_get_org_by_id("6348e8efac2afce6962bf2e1"), to_scores=35, childs=0, number=202)
sn_step_parent(sn_get_org_by_id("6348eb24ac2afce6962bf2e3"), to_scores=35, childs=0, number=203)
sn_step_parent(sn_get_org_by_id("63490169ac2afce6962bf2f8"), to_scores=35, childs=0, number=204)
sn_step_parent(sn_get_org_by_id("6349053cac2afce6962bf2fc"), to_scores=35, childs=0, number=205)
sn_step_parent(sn_get_org_by_id("63490a77ac2afce6962bf303"), to_scores=35, childs=0, number=206)
sn_step_parent(sn_get_org_by_id("63490f59ac2afce6962bf308"), to_scores=35, childs=0, number=207)
sn_step_parent(sn_get_org_by_id("63491938ac2afce6962bf311"), to_scores=35, childs=0, number=208)
sn_step_parent(sn_get_org_by_id("63491297ac2afce6962bf30b"), to_scores=35, childs=0, number=209)
sn_step_parent(sn_get_org_by_id("63491a6dac2afce6962bf313"), to_scores=35, childs=0, number=210)
sn_step_parent(sn_get_org_by_id("63491b11ac2afce6962bf314"), to_scores=35, childs=0, number=211)
sn_step_parent(sn_get_org_by_id("63491be3ac2afce6962bf315"), to_scores=35, childs=0, number=212)
sn_step_parent(sn_get_org_by_id("634a79437df062bb7803da0d"), to_scores=35, childs=0, number=213)
sn_step_parent(sn_get_org_by_id("634a886909dfb99f5c6aa066"), to_scores=35, childs=0, number=214)
sn_step_parent(sn_get_org_by_id("634a79f47df062bb7803da0e"), to_scores=35, childs=0, number=215)
sn_step_parent(sn_get_org_by_id("634a917909dfb99f5c6aa073"), to_scores=35, childs=0, number=216)
sn_step_parent(sn_get_org_by_id("634a950909dfb99f5c6aa077"), to_scores=35, childs=0, number=217)
sn_step_parent(sn_get_org_by_id("634a95eb09dfb99f5c6aa078"), to_scores=35, childs=0, number=218)
sn_step_parent(sn_get_org_by_id("634a97f909dfb99f5c6aa07a"), to_scores=35, childs=0, number=219)
sn_step_parent(sn_get_org_by_id("634aa33009dfb99f5c6aa085"), to_scores=35, childs=0, number=220)
sn_step_parent(sn_get_org_by_id("634aa42009dfb99f5c6aa086"), to_scores=35, childs=0, number=221)
sn_step_parent(sn_get_org_by_id("634ab11309dfb99f5c6aa093"), to_scores=35, childs=0, number=222)
sn_step_parent(sn_get_org_by_id("634b0507b948a21cb96d6761"), to_scores=35, childs=0, number=223)
sn_step_parent(sn_get_org_by_id("634b086961f224d31bbd371d"), to_scores=35, childs=0, number=224)
sn_step_parent(sn_get_org_by_id("634b0a1b8b08045593dd0745"), to_scores=35, childs=0, number=225)
sn_step_parent(sn_get_org_by_id("634b0cc24cc8e3ab21f95b1f"), to_scores=35, childs=0, number=226)
sn_step_parent(sn_get_org_by_id("634b8a7395fd4b6edda478e4"), to_scores=35, childs=0, number=227)
sn_step_parent(sn_get_org_by_id("634b2ce5795b7617b831d856"), to_scores=35, childs=0, number=228)
sn_step_parent(sn_get_org_by_id("634bb8c3eb54d82adca4c8a3"), to_scores=35, childs=0, number=229)
sn_step_parent(sn_get_org_by_id("634bba86eb54d82adca4c8d0"), to_scores=35, childs=0, number=230)
sn_step_parent(sn_get_org_by_id("634a5f219742ae236369bbbd"), to_scores=35, childs=0, number=231)
sn_step_parent(sn_get_org_by_id("634a5ecc9742ae236369bbbc"), to_scores=35, childs=0, number=232)
sn_step_parent(sn_get_org_by_id("634bd1b6c6f5ae1320a7be77"), to_scores=35, childs=0, number=233)
sn_step_parent(sn_get_org_by_id("634bd29ac6f5ae1320a7be78"), to_scores=35, childs=0, number=234)
sn_step_parent(sn_get_org_by_id("634c2856f83c017d34439acf"), to_scores=35, childs=0, number=235)
sn_step_parent(sn_get_org_by_id("634c2bcaf83c017d34439b74"), to_scores=35, childs=0, number=236)
sn_step_parent(sn_get_org_by_id("634c3beb94b587fc2cbffb2a"), to_scores=35, childs=0, number=237)
sn_step_parent(sn_get_org_by_id("634c40b094b587fc2cbffb2d"), to_scores=35, childs=0, number=238)
sn_step_parent(sn_get_org_by_id("634c4a1a127623944a50401f"), to_scores=35, childs=0, number=239)
sn_step_parent(sn_get_org_by_id("634c4e4f127623944a504024"), to_scores=35, childs=0, number=240)
sn_step_parent(sn_get_org_by_id("634c5093127623944a504027"), to_scores=35, childs=0, number=241)
sn_step_parent(sn_get_org_by_id("634c54699b83cea15a2b9eaa"), to_scores=35, childs=0, number=242)
sn_step_parent(sn_get_org_by_id("634c55489b83cea15a2b9eab"), to_scores=35, childs=0, number=243)
sn_step_parent(sn_get_org_by_id("634c56139b83cea15a2b9eac"), to_scores=35, childs=0, number=244)
sn_step_parent(sn_get_org_by_id("634c56df9b83cea15a2b9ead"), to_scores=35, childs=0, number=245)
sn_step_parent(sn_get_org_by_id("634c58c09b83cea15a2b9eaf"), to_scores=35, childs=0, number=246)
sn_step_parent(sn_get_org_by_id("634c5a479b83cea15a2b9eb1"), to_scores=35, childs=0, number=247)
sn_step_parent(sn_get_org_by_id("634c5d3b9b83cea15a2b9eb4"), to_scores=35, childs=0, number=248)
sn_step_parent(sn_get_org_by_id("634d06a85f8bfcffdaddc469"), to_scores=35, childs=0, number=249)
sn_step_parent(sn_get_org_by_id("634e68881d4ff37def5565f5"), to_scores=35, childs=0, number=250)



quit()

num = 0
for org in population.get_all():
    if str(org.id) in (
#"634e58201d4ff37def5565ea",
"632ad2429dbf8ba893b5e25a",
):
        num = num + 1
        sn_step_parent(org, to_scores=89, childs=0, number=num)

quit()

















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
#"634934c1ac2afce6962bf327",
#"634936f2ac2afce6962bf329",
#"634954e2a5657a22c93bcb1f",
#"6349817ba5657a22c93bcb3b",
#"63499296a5657a22c93bcb48",
#"634998dda5657a22c93bcb4c",
#"6349a280a5657a22c93bcb52",
#"6349aee2a5657a22c93bcb5a",
#"6349bd64a5657a22c93bcb63",
#"634a100c70b6e7de108df9eb",
#"634a116070b6e7de108df9ec",
#"634a32f870b6e7de108df9f5",


"6345eec9636053dd92556e16",
"6347b27bac2afce6962bf1a7",
"6348adc0ac2afce6962bf2b7",
"6348be45ac2afce6962bf2c9",
"6348beacac2afce6962bf2ca",
"6348fd98ac2afce6962bf2f4",
"63499296a5657a22c93bcb48",



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
#    _end = d_max or listing.all_history_date(_tickers)[-1]
    _end = d_max or all_history_date(_tickers)[-1]

#dates = listing.all_history_date(_tickers, start=_end)
dates = all_history_date(_tickers, start=_end)
if (d_min != _end) or (len(dates) == 1):
    step = step + 0
#    print(step + 1)

else:
    _end = dates[1]

#print(_tickers, _end, step)
##################


##org = population.get_next_one()

for org in population.get_all():
    if 1==1 and str(org.id) in (

#"63377659f23069f3da7a45df",
#"6338c18be8249177d2e3eaa3",
#"6338c3a5e8249177d2e3eaa4",
#"633b087ee8249177d2e3eae5",
#"633ef9b0d88bc0499b8426c1",
#"634075a283329d94dcefdefc"

):
         print("Working with ", org.id)
         sn_step(org)				# Обучить эту модель на один день.
quit()

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


## margin calc  to manual clear
p_metrics = []

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




#dates = listing.all_history_date(_tickers)
dates = all_history_date(_tickers)
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


#print(f"Num\tID\tcrDate\tcrTime\tallRETs")   ##DATE добавить
print(
    "\t".join(
        [
            f"Num",
            f"ID",
            f"crDate",
            f"crTime",
            f"",
            f"",
            f"",
            f"",
            f"",
            f"allRETs",
        ],
    ),
)

#print(f"Num\tID\tcrDate\tcrTime\tallLLHs")
print(
    "\t".join(
        [
            f"Num",
            f"ID",
            f"crDate",
            f"crTime",
            f"",
            f"",
            f"",
            f"",
            f"",
            f"allLLHs",
        ],
    ),
)

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
#            self._end = d_max or listing.all_history_date(self._tickers)[-1]
            self._end = d_max or all_history_date(self._tickers)[-1]

#        dates = listing.all_history_date(self._tickers, start=self._end)
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

#        all_dates = listing.all_history_date(self._tickers, end=self._end)
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
