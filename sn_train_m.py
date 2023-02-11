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

import time


import logging
logger = logging.getLogger()




#population.print_stat()

from poptimizer.evolve import store

import logging
LOGGER = logging.getLogger()

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


def sn_get_margin(org: population.Organism, min_margin : float = 0) -> tuple[float, float]:
    import logging
    logger = logging.getLogger()

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

    logger.info(f"Margin - {margin:.2%}, Slowness - {time_score:.2%}\n")  # noqa: WPS221

    if margin < min_margin:
        logger.info(f"{org.id}  Исключен из популяции (min_margin = {min_margin})...\n")
        org.die()
#    else:
#        logger.info(f"{org.id}  НЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕЕ Исключен из популяции...\n")

    return margin, time_score



def sn_tests(tsts : int = 0) -> float:
    if tsts != 0:
        return tsts

#    logger.info(f"!!!!!!!!!!!!! stndart")

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


def sn_eval_organism(organism: population.Organism, tsts : int = 0, min_margin : float = 0) -> Optional[tuple[float, float]]:

    """Оценка организмов.

    - Если организм уже оценен для данной даты, то он не оценивается.
    - Если организм старый, то оценивается один раз.
    - Если организм новый, то он оценивается для определенного количества дат из истории.
    """
    try:
        logger.info(f"{organism}\n")
    except AttributeError as err:
        organism.die()
        logger.info(f"Удаляю - {err}\n")

        return None

    all_dates = listing.all_history_date(_tickers, end=_end)
#    dates = all_dates[-sn_tests :].tolist()
    tmpsn = sn_tests(tsts)        				# кол-во тестов.
    dates = all_dates[-tmpsn :].tolist()		# торговые даты от самой старой (слева) до самой новой (справа).  На кол-во тестов

#    print(f"!!!!!!!!!!!! tmpsn={tmpsn}, dates={dates}, _end={_end}, organism.date={organism.date}, organism.scores={organism.scores}, population.max_scores={population.max_scores()}")
#   _end   # - (последний торговый день (ближайшая к текущей торговая дата назад)
#   organism.date  # Последняя дата тестирования организма (он тестируется от старой даты к текущей)
#   organism.scores   # Сколько тестов прошел организм.
#   population.max_scores()   # максимальный wins популяции

    if organism.date == _end and (sn_tests(tsts) < population.max_scores() or organism.scores == sn_tests(tsts) - 1):	## На один день в прошлое
        dates = [all_dates[-(organism.scores + 1)]]
        logger.info(f"!!!!!!!!!!!! 11111")
    elif organism.date == _end and sn_tests(tsts) >= population.max_scores() and organism.scores < sn_tests(tsts) - 1:
        organism.clear()
        logger.info(f"!!!!!!!!!!!! 22222")
    elif organism.scores:              # Если новый алгоритм - у него пустой scores. Оценивает диапазон на текущую дату.   Остановленный на этапе тестирования организм    начинает отсюда и возможно ломает
        dates = [_end]
        logger.info(f"!!!!!!!!!!!! 33333")



    cnt = 0
    for date in dates:
        cnt += 1
        logger.info(f"!!!!!! date={date}   {cnt} of {len(dates)}")
        try:
            organism.evaluate_fitness(_tickers, date, sn_comments = f"{organism.id}\t{date}")
        except (ModelError, AttributeError) as error:
            organism.die()
            logger.info(f"Удаляю - {error}\n")

            return None

    return sn_get_margin(organism, min_margin)
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
    _end = d_max or listing.all_history_date(_tickers)[-1]

    all_dates = listing.all_history_date(_tickers, end=_end)
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


def sn_make_child(hunter: population.Organism) -> Optional[population.Organism]:
    """Один random child."""
    logger.info(f"!!! Родитель {hunter.id} (Scale={sn_scale():.2f}):")

    hunter = hunter.make_child(1 / sn_scale())

    if (margin := sn_eval_organism(hunter)) is None:
        return None
    if margin[0] < 0:
        return None



def sn_get_org_by_id(str_org: str) -> population.Organism:
    for org in population.get_all():
        if str(org.id) == str_org:
            return org

import random
def sn_get_parents(*str_org: str, allow_mono: str):   
# Возвращает троих родителей, выбирая их из поданных на вход строк
# Позволять моногамность (0 нет; 1 - пара родителей12 могут пересекаться; 2 - все могут пересекаться)?
    if len(str_org) == 0:     # Если на вход не подано объектов для выбора, то выходим
        return ""

    if allow_mono == 2 or len(str_org) <= 2:     # Если на вход подано 2 или меньше объектов - то могут пересекаться как хотят
        return  str_org[random.randint(0, len(str_org)-1)], \
                str_org[random.randint(0, len(str_org)-1)], \
                str_org[random.randint(0, len(str_org)-1)]
    if allow_mono == 1:
        par_2 = str_org[random.randint(0, len(str_org)-1)]
        par_1 = str_org[random.randint(0, len(str_org)-1)]
        par_0 = str_org[random.randint(0, len(str_org)-1)]
        while par_0 == par_1 or par_0 == par_2:
            par_0 = str_org[random.randint(0, len(str_org)-1)]
        return(par_0, par_1, par_2)
    if allow_mono == 0:
        par_2 = str_org[random.randint(0, len(str_org)-1)]
        par_1 = str_org[random.randint(0, len(str_org)-1)]
        while par_1 == par_2:
            par_1 = str_org[random.randint(0, len(str_org)-1)]

        par_0 = str_org[random.randint(0, len(str_org)-1)]
        while par_0 == par_1 or par_0 == par_2:
            par_0 = str_org[random.randint(0, len(str_org)-1)]
        return(par_0, par_1, par_2)


def sn_make_child_complex(
# Функция для рождения ребенка
    *str_org: str,                  # Родитель или набор родителей для рандомного выбора
    steps_type: str = 'parent',     # steps_type ('parent' - как у родителя + steps_val, 
#                                      'max'/'min' - максимальное/минимальное в популяции + steps_val, 
#                                      'exactly' - ровно steps_val
    steps_val: int = 1,
    allow_mono: int = 0,            # Позволять моногамность (0 нет; 1 - пара родителей12 могут пересекаться; 2 - все могут пересекаться)?
    min_margin_save: float = 0.0    # минимальный маржин меньше и равно которому не сохранять
):
    par0, par1, par2 = sn_get_parents(*str_org, allow_mono = allow_mono)
    par_0 = sn_get_org_by_id(par0)
    par_1 = sn_get_org_by_id(par1)
    par_2 = sn_get_org_by_id(par2)


    if steps_type == 'parent':                #  Доделать
        s_tsts = s_tsts + steps_val
    elif steps_type == 'standart':
        s_tsts = sn_tests() + steps_val
    elif steps_type == 'max':
        s_tsts = population.max_scores() + steps_val
    elif steps_type == 'min':
        s_tsts = population.min_scores() + steps_val
    else:
        s_tsts = steps_val


    logger.info(f"!!! BEG Родители: {par0}, {par1}, {par2}; (Scale={sn_scale():.2f}); Тестов({steps_type}): {s_tsts}, min_margin: {min_margin_save}")

    hunter = par_0.make_child(1 / sn_scale(), par1 = par_1, par2 = par_2)


    if (margin := sn_eval_organism(hunter, tsts = s_tsts, min_margin = min_margin_save / 100)) is None:
        logger.info(f"!!! END_KILLED_1 Родители: {par0}, {par1}, {par2}; (Scale={sn_scale():.2f}); Тестов({steps_type}): {s_tsts}")
        return None
    if margin[0] < min_margin_save / 100:
        logger.info(f"!!! END_KILLED_2 Родители: {par0}, {par1}, {par2}; (Scale={sn_scale():.2f}); Тестов({steps_type}): {s_tsts}; margin: {margin[0]:.2%}")
        return None

    logger.info(f"!!! END_SAVED Родители: {par0}, {par1}, {par2}; (Scale={sn_scale():.2f}); Тестов({steps_type}): {s_tsts}; margin: {margin[0]}:.2%")





#####################

collection = store.get_collection()





_tickers = None
_end = None


step = 1
org = None

d_min, d_max = population.min_max_date()
if _tickers is None:
    _tickers = load_tickers()
    _end = d_max or listing.all_history_date(_tickers)[-1]

dates = listing.all_history_date(_tickers, start=_end)
if (d_min != _end) or (len(dates) == 1):
    step = step + 0
else:
    _end = dates[1]


cnt=0

while 1==1:
    try:
        cnt=cnt+1
        logger.info(f"!!! Counter: {cnt}")

        sn_make_child_complex(
#"636fa9895794666f40ab4f80",
#"6371128b5794666f40ab4fca",
#"6387096a19d21242e5f5a5d5",
#"638bdad40688c848b0fb3d7b",
#"6392031f282e87fcc7f2d6f6",
#"6395c2c20b63a2803ab23700",
#"6395cb960b63a2803ab23702",
#"639f0e9a1673dfd7c07fb0cb",

#Хорошо:
#"6366d526a9dd86830235ca5e",
#"636fa9895794666f40ab4f80",
#"6371128b5794666f40ab4fca",
#"637fa520b171fb5c77f484d8",
#"6387096a19d21242e5f5a5d5",
#"638ac7200688c848b0fb3d48",
#"638b7a7e0688c848b0fb3d6a",
#"638bdad40688c848b0fb3d7b",
#"6392031f282e87fcc7f2d6f6",
#"6395c2c20b63a2803ab23700",
#"6395cb960b63a2803ab23702",
#"6397880136c8206eaa6dfd5a",
#"639e4ed51673dfd7c07fb0ae",
#"639ea4e81673dfd7c07fb0bb",
#"639f0e9a1673dfd7c07fb0cb",


#Плохо  0 за 12 часов:
#"636fa9895794666f40ab4f80",
#"638209e9da4c16eb435503d0",
#"6387096a19d21242e5f5a5d5",
#"638b7a7e0688c848b0fb3d6a",
#"638bdad40688c848b0fb3d7b",
#"638e266762f78314766d9543",
#"639678cf70364b20e3f9deb2",

# 5/29 за 14 часов
# 6/36 за 15 часов

#"6366d526a9dd86830235ca5e",
#"636fa9895794666f40ab4f80",
#"6371128b5794666f40ab4fca",
#"63746e45d9c43d3272d528ed",
#"6380705c380fdee0ca210cc1",
#"6382288bda4c16eb435503d3",
#"6387096a19d21242e5f5a5d5",
#"6389fee7e7dd9183b6d61c5a",
#"638ac7200688c848b0fb3d48",
#"638f4984962eaf85e31f3e4e",
#"6392031f282e87fcc7f2d6f6",
#"63921aaf626eaa626a09b97d",
#"6392c0e333f64191a78530d5",
#"63942c2d3bbfe4dbb461b47d",
#"6395c2c20b63a2803ab23700",
#"6395cb960b63a2803ab23702",
#"639978bd9105ee7479ffefc9",
#"639aa3427a3fb7cf4a1dec46",
#"639dc6051673dfd7c07fb099",
#"639f0e9a1673dfd7c07fb0cb",
#"639f52a41673dfd7c07fb0d6",
#"63a10101d56a19c7da0eb2b0",
#"63a22779e084281b2f79dfe0",


#301  at 23.00 23/12
#312  at 12:53 24/12
#"638d8f8562f78314766d9529",
#"6358959fdc145b8a766c035c",
#"6366d526a9dd86830235ca5e",
#"636fa9895794666f40ab4f80",
#"6371128b5794666f40ab4fca",
#"63746e45d9c43d3272d528ed",
#"637fa520b171fb5c77f484d8",
#"6382288bda4c16eb435503d3",
#"6387096a19d21242e5f5a5d5",
#"638ac7200688c848b0fb3d48",
#"638b7a7e0688c848b0fb3d6a",
#"638e266762f78314766d9543",
#"6392031f282e87fcc7f2d6f6",
#"63942c2d3bbfe4dbb461b47d",
#"6395c2c20b63a2803ab23700",
#"6395cb960b63a2803ab23702",
#"6396c0e00941f8f33c0851d8",
#"639f0a371673dfd7c07fb0ca",
#"639f0e9a1673dfd7c07fb0cb",
#"639f52a41673dfd7c07fb0d6",
#"63a020264883d5b93afeef18",
#"63a0fae8d56a19c7da0eb2af",
#"63a22779e084281b2f79dfe0",
#"63a27817e084281b2f79dfed",
#"63a4603f0a08752bc956e714",
#"63a4c225fe9379b3d6d8947e",
#"63a541df39248539df67cd59",
#"63a5aa4939248539df67cd67",
#"63a5e33e39248539df67cd71",


#"6358959fdc145b8a766c035c",#	margin>0.4
#"636fa9895794666f40ab4f80",#	ret-top5, margin-middle
#"6387096a19d21242e5f5a5d5",#	ret-top5; margin>0.4
#"638a4be3e82fe0c3a70b55bd",#	ret-top5; 
#"638ac7200688c848b0fb3d48",#	llhup>0, llhmed>0
#"638b7a7e0688c848b0fb3d6a",#	margin>0.4
#"638d8f8562f78314766d9529",#	ret-top5; 
#"638da49262f78314766d952d",#	LLHmax-max
#"638f4fe5962eaf85e31f3e4f",#	retup>0, retmed>=0; margin>0.4
#"6395c2c20b63a2803ab23700",#	margin>0.4
#"6395cb960b63a2803ab23702",#	retup>0, retmed>=0; 
#"6395ed84f3ea4baebdad0e3c",#	margin>0.4
#"639dc6051673dfd7c07fb099",#	retupp-max
#"639e7d2c1673dfd7c07fb0b5",#	margin>0.4
#"639fc86b4883d5b93afeef09",#	margin>0.4
#"63a10101d56a19c7da0eb2b0",#	margin>0.4
#"63a22779e084281b2f79dfe0",#	retup>0, retmed>=0; 
#"63a4603f0a08752bc956e714",#	retup>0, retmed>=0; margin>0.4
#"63a57f2239248539df67cd61",#	retup>0, retmed>=0; margin>0.4
#"63a5a24639248539df67cd66",#	retupp-max
#"63a636c44733e44f023d1a04",#	margin>0.4
#"63a6710a4733e44f023d1a0d",#	LLH-top5, margin-middle
#"63a6764a4733e44f023d1a0e",#	LLHmax-max
#"63a73f91e3734a9dc39e6453",#	retupp-max
#"63a8d52c4fdb6a0d64b52f31",#	margin>0.4
#"63a9b30150f2503cd6e8336d",#	LLH-top5, margin-middle
#"63aa8ccd3242e63872d5c8aa",#	retupp-max
#"63ab9b9a3242e63872d5c8d2",#	margin>0.4
#"63ac8673032f462775168104",#	retup>0, retmed>=0; 
#"63af0b54c7b33e28936e4460",#	retup>0, retmed>=0; 
#"63af51f7c7b33e28936e4465",#	LLHmax-max
#"63b0096fd4492e5d8d71d95b",#	LLHmax-max
#"63b0c78dd4492e5d8d71d974",#	margin>0.4
#"63b0da7ed4492e5d8d71d976",#	LLH-top5, margin-middle

#"638ac7200688c848b0fb3d48",#	llhup>0, llhmed>0
#"638b7a7e0688c848b0fb3d6a",#	retup>0, retmed>=0; margin>0.4, RET-good
#"638d8f8562f78314766d9529",#	ret-good, margin-middle
#"638f4fe5962eaf85e31f3e4f",#	margin>0.4, RET-good
#"639008c1962eaf85e31f3e6c",#	ret-good, margin-middle
#"6395cb960b63a2803ab23702",#	retup>0, retmed>=0; 
#"6396c0e00941f8f33c0851d8",#	ret-good, margin-middle
#"639e7d2c1673dfd7c07fb0b5",#	LLH-top5, margin-middle; RET-top2
#"639f0e9a1673dfd7c07fb0cb",#	LLH-top5, margin-middle; RET-top2
#"63a10101d56a19c7da0eb2b0",#	retup>0, retmed>=0; 
#"63a4603f0a08752bc956e714",#	ret-good, margin-middle
#"63a57f2239248539df67cd61",#	retup>0, retmed>=0; margin>0.4, RET-good
#"63a636c44733e44f023d1a04",#	retup>0, retmed>=0; margin>0.4, RET-good
#"63a6764a4733e44f023d1a0e",#	ret-good, margin-middle
#"63aa8ccd3242e63872d5c8aa",#	retupp-top3
#"63ab9b9a3242e63872d5c8d2",#	retupp-top3
#"63ac8673032f462775168104",#	retup>0, retmed>=0; 
#"63af51f7c7b33e28936e4465",#	margin>0.4, RET-good
#"63b0c78dd4492e5d8d71d974",#	margin>0.4, RET-good
#"63b1eaf14b30ead81033086f",#	retup>0, retmed>=0; margin>0.4, RET-good
#"63b21f8a4b30ead810330876",#	retupp-top3
#"63b281574b30ead810330884",#	retupp-top3
#"63b339201cfb7a8924c8bf08",#	retupp-top3
#"63b3c4c50aa240f5c9ad457c",#	margin>0.4, RET-good

#"638b7a7e0688c848b0fb3d6a",#	margin>=0, RET-good
#"63942c2d3bbfe4dbb461b47d",#	jj
#"6395cb960b63a2803ab23702",#	margin>=0, RET-good
#"6395ed84f3ea4baebdad0e3c",#	margin>=0.3, RET-good
#"639e7d2c1673dfd7c07fb0b5",#	margin>=0, RET-good
#"639f0e9a1673dfd7c07fb0cb",#	margin>=0, RET-good
#"639fc86b4883d5b93afeef09",#	ff
#"63a5a24639248539df67cd66",#	jj
#"63a9b30150f2503cd6e8336d",#	jj
#"63ab9b9a3242e63872d5c8d2",#	jj
#"63aba3773242e63872d5c8d3",#	ff
#"63af51f7c7b33e28936e4465",#	margin>=0.3, RET-good
#"63b0da7ed4492e5d8d71d976",#	jj
#"63b192a7d4492e5d8d71d98a",#	jj
#"63b281574b30ead810330884",#	jj
#"63b2b0e14b30ead81033088a",#	ff
#"63b2c12e4b30ead81033088c",#	margin>=0.3, RET-good
#"63b2e7a34b30ead810330891",#	jj
#"63b3c4c50aa240f5c9ad457c",#	margin>=0.3, RET-good
#"63b3fd9bee42aa39a2dea19b",#	ff
#"63b413acee42aa39a2dea19e",#	margin>=0.3, RET-good
#"63b4758dee42aa39a2dea1aa",#	margin>=0, RET-good
#"63b4f85aee42aa39a2dea1c1",#	jj
#"63b50151ee42aa39a2dea1c2",#	margin>=0.3, RET-good
#"63b57a7306e4f9616ef051db",#	margin>=0, RET-good
#"63b582cf06e4f9616ef051dc",#	margin>=0.3, RET-good
#"63b602507e59a233ace2bc09",#	margin>=0, RET-good
#"63b621677e59a233ace2bc0d",#	ff
#"63b656f536fa20111430c2bd",#	jj


"63942c2d3bbfe4dbb461b47d",
"63aa8ccd3242e63872d5c8aa",
"63aba3773242e63872d5c8d3",
"63b0da7ed4492e5d8d71d976",
"63b21f8a4b30ead810330876",
"63b281574b30ead810330884",
"63b4f85aee42aa39a2dea1c1",
"63b50151ee42aa39a2dea1c2",
"63b656f536fa20111430c2bd",
"63b6ee323beee2cc8cc46506",
"63b733b43beee2cc8cc4650f",
"63b7918d3beee2cc8cc4651c",





steps_type = 'max',        # steps_type ('parent' - как у родителя + steps_val, 
#steps_val = 1,
steps_val = random.randint(0, 9) * -1,
#steps_type = 'standart',        # steps_type ('parent' - как у родителя + steps_val, 
#steps_val = 1,


allow_mono = 0,            # Позволять моногамность (0 нет; 1 - пара родителей12 могут пересекаться; 2 - все могут пересекаться)?
min_margin_save = 0.05     # минимальный маржин меньше и равно которому не сохранять
)

        time.sleep(1)
    except:
        print("error\n")
        time.sleep(3)


#sn_step_parent(sn_get_org_by_id("635189748293e7ae35191190"), to_scores=50, childs=0, number=1)
#sn_step_parent(sn_get_org_by_id("635189748293e7ae35191194"), to_scores=50, childs=0, number=2)
#sn_step_parent(sn_get_org_by_id("635189748293e7ae35191199"), to_scores=50, childs=0, number=3)



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



