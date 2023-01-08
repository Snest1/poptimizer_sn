"""Эволюция параметров модели."""
import datetime
import itertools
import logging
import operator
from typing import Optional

import numpy as np
from scipy import stats

from poptimizer import config
from poptimizer.data.views import listing
from poptimizer.dl import ModelError
from poptimizer.evolve import population, seq
from poptimizer.portfolio.portfolio import load_tickers


def LoadFromTV(ticker: str, TVformula: str, mult: int, sn_dates: list):
## Тут загружаем в базу quotes все нужные данные для тех бумаг, готорый нет в TQBR
        from poptimizer.shared import adapters, col
        import pandas as pd

        sn_ticker = ticker
        sn_df_quotes = pd.DataFrame(columns = [col.DATE, col.OPEN, col.CLOSE, col.HIGH, col.LOW, col.TURNOVER])
        sn_df_quotes = sn_df_quotes.set_index(col.DATE)

#        import datetime
#        bars = 10
#        datetime.datetime.strptime(last_date, "%Y-%m-%d").date() - datetime.datetime.strptime(start_date, "%Y-%m-%d").date()

        from poptimizer.tvDatafeed import TvDatafeed,Interval
        tv = TvDatafeed()
        tv.clear_cache()
#        t_GLDRUB = tv.get_hist("MOEX:GLDRUB_TOM", interval=Interval.in_daily,n_bars=bars.days)
        t_GLDRUB = tv.get_hist(TVformula, interval=Interval.in_daily,n_bars=100000)
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
        return population.max_scores() ** 0.5

    @property
    def _tests(self) -> int:
        return population.count()

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
            for i in range(1, config.START_POPULATION):
                self._logger.info(f"Создается базовый организм {i}:")
                org = population.create_new_organism()
                self._logger.info(f"{org}\n")

    def _step(self, hunter: population.Organism) -> Optional[population.Organism]:
        """Один шаг эволюции."""
        have_more_dates = hunter.date and self._end > hunter.date

        label = ""
        if not hunter.scores:
            label = " - новый организм"

        self._logger.info(f"Родитель{label} (ID={hunter.id}):")
        if self._eval_organism(hunter) is None:
            return None

        if have_more_dates:
            self._logger.info("Появились новые данные - не размножается...\n")

            return None

        for n_child in itertools.count(1):
            self._logger.info(f"Потомок {n_child} (Scale={self._scale:.2f}):")

            hunter = hunter.make_child(1 / self._scale)
            if (margin := self._eval_organism(hunter)) is None:
                return None

            if (rnd := np.random.random()) < (slowness := margin[1]):
                self._logger.info(f"2 Медленный, не размножается (но будет сохранен) {rnd=:.2%} < {slowness=:.2%}...\n")
                return None

    def _eval_organism(self, organism: population.Organism) -> tuple[float, float] | None:
        try:
            self._logger.info(f"{organism}\n")
        except AttributeError as err:
            self._logger.error(f"!!!_  Organizm:{organism.id}  Проблема: 04 Удаляю - {err}\n")
            organism.die()
#            self._logger.error(f"Удаляю - {err}\n")

            return None

        all_dates = listing.all_history_date(self._tickers, end=self._end)
        sn_len = 0

        try:
            if organism.date == self._end :
                prob = 1 - _time_delta(organism)
                retry = stats.geom.rvs(prob)
                dates = all_dates[-(max(organism.scores, self._tests - 1)  + retry): -organism.scores].tolist()
                organism.retrain(self._tickers, dates[0], sn_comments = f"{organism.id}\t{dates[0]}"))
                sn_len = len(dates)
                dates = reversed(dates)
            elif organism.scores:
                if self._tickers != tuple(organism.tickers):
                    organism.retrain(self._tickers, self._end, sn_comments = f"{organism.id}\t{self._end}")
                dates = [self._end]
                sn_len = len(dates)
            else:
                dates = all_dates[-self._tests :].tolist()
                sn_len = len(dates)
                organism.retrain(self._tickers, dates[0], sn_comments = f"{organism.id}\t{dates[0]}")
        except (ModelError, AttributeError) as error:
            self._logger.error(f"!!!_ Organizm:{organism.id}  Проблема: 09 Удаляю - {error}\n")
            organism.die()
#            self._logger.error(f"Удаляю - {error}\n")

            return None


        cnt = 0
        for date in dates:
#            print(f"!!!!!! date={date} in dates={dates}")
            cnt += 1
            self._logger.info(f"!!!!!! date={date}   {cnt} of {sn_len}")
            try:
                organism.evaluate_fitness(self._tickers, date, sn_comments = f"{organism.id}\t{date}")
            except (ModelError, AttributeError) as error:
                self._logger.error(f"!!!_ Organizm:{organism.id}  Проблема: 03 Удаляю - {error}\n")
                organism.die()
#                self._logger.error(f"Удаляю - {error}\n")

                return None

        return self._get_margin(organism)

    def _get_margin(self, org: population.Organism) -> tuple[float, float] | None:
        """Используется тестирование разницы llh и ret против самого старого организма.

        Используются тесты для связанных выборок, поэтому предварительно происходит выравнивание по
        датам и отбрасывание значений не имеющих пары (возможно первое значение и хвост из старых
        значений более старого организма).
        """
        names = {"llh": "LLH", "ir": "RET"}
        upper_bound = 1

        for metric in ("ir", "llh"):
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

            if upper < 0:
                self._logger.error(f"!!!_                          Organizm:{organism.id}  Проблема: 08 Удаляю - {error}\n")
                org.die()
                self._logger.info("Исключен из популяции...\n")

                return None

            upper_bound *= upper ** 0.5

        org.upper_bound = upper_bound
        time_score = _time_delta(org)

        self._logger.info(f"Upper bound - {upper_bound:.4f}, Slowness - {time_score:.2%}\n")  # noqa: WPS221

        return upper_bound, time_score


def _time_delta(org):
    times = [doc["timer"] for doc in population.get_metrics() if "timer" in doc]

    return stats.percentileofscore(times, org.timer, kind="mean") / 100


def _check_time_range(self) -> bool:

### Если ранее былии загружены курсы MOEX, надо обновить остальные курсы  2023

    from pymongo.collection import Collection
    from poptimizer.store.database import DB, MONGO_CLIENT
    misc_collection = MONGO_CLIENT[DB]['misc']

    if (misc_collection.find_one({'_id': "need_update_TV"})):
        print("!!!!!!!!!!!FOUNDED!!!!!!!!!!!!!!")
#        quit()


#  загрузим список тикеров и дат, т.к. другие библиотеки poptimizer не работают
        from pymongo import MongoClient
        quotes_collection = MongoClient('localhost', 27017)['data']['quotes']
        sn_tickers = []   # список
        sn_dates = []   # список
        for quote in quotes_collection.find():
            if quote['_id'] == 'GAZP':
                sn_tickers.append(quote['_id'])
                for one_date in quote['data']['index']:
                    if one_date not in sn_dates:
                        sn_dates.append(one_date)
        sn_dates.sort()
        print(sn_dates)


        LoadFromTV('GLDRUB_TOM', "MOEX:GLDRUB_TOM", 1, sn_dates)
        LoadFromTV('SLVRUB_TOM', "MOEX:SLVRUB_TOM", 1, sn_dates)
        LoadFromTV('BTCRUB', "BINANCE:BTCRUB/10000", 10000, sn_dates)
        LoadFromTV('ETHRUB', "BINANCE:ETHRUB/100", 100, sn_dates)

        misc_collection.delete_one({'_id': "need_update_TV"})

#    quit()
############################################################
##  тут еще бы сделать проверку, что не загрузилось лишнего
################################



# Запуск различных заданий

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
