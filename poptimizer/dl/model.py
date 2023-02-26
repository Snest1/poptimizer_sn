"""Тренировка модели."""
import collections
import io
import itertools
import logging
import sys
from typing import Final, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from scipy import optimize
from torch import nn, optim

from poptimizer import config
from poptimizer.config import DEVICE, YEAR_IN_TRADING_DAYS
from poptimizer.dl import data_loader, ledoit_wolf, models, PhenotypeData
from poptimizer.dl.features import data_params
from poptimizer.dl.forecast import Forecast
from poptimizer.dl.models.wave_net import GradientsError, ModelError


# Ограничение на максимальное снижение правдоподобия во время обучения для его прерывания
LLH_DRAW_DOWN = 1

# Максимальный размер документа в MongoDB
MAX_DOC_SIZE: Final = 2 * (2**10) ** 2

# Максимальный размер батча GB
MAX_BATCH_SIZE: Final = 150

DAY_IN_SECONDS: Final = 24 * 60**2

LOGGER = logging.getLogger()

from pymongo.collection import Collection
from poptimizer.store.database import DB, MONGO_CLIENT
_COLLECTION_speedy = MONGO_CLIENT[DB]["sn_speedy"]
def snspeedy_get_collection() -> Collection:
    return _COLLECTION_speedy




class TooLongHistoryError(ModelError):
    """Слишком длинная история признаков.

    Отсутствуют история для всех тикеров - нужно сократить историю.
    """


class TooLargeModelError(ModelError):
    """Слишком большая модель.

    Модель с 2 млн параметров не может быть сохранена.
    """


class DegeneratedModelError(ModelError):
    """В модели отключены все признаки."""


def log_normal_llh_mix(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minus Normal Log Likelihood and forecast means."""
    dist = model.dist(batch)
#    llh = dist.log_prob(batch["Label"] + torch.tensor(1.0))	#ошибка при очень длинных тестах

    try:
        llh = dist.log_prob(batch["Label"] + torch.tensor(1.0))
    except ValueError:
        raise GradientsError(f"Wrong bound in Categorical distribution")

    return -llh.sum(), dist.mean - torch.tensor(1.0), dist.variance


class Model:
    """Тренирует, тестирует и прогнозирует модель на основе нейронной сети."""

    def __init__(
        self,
        tickers: tuple[str, ...],
        end: pd.Timestamp,
        phenotype: data_loader.PhenotypeData,
        pickled_model: Optional[bytes] = None,
        sn_comments: str = "",
    ):
        """Сохраняет необходимые данные.

        :param tickers:
            Набор тикеров для создания данных.
        :param end:
            Конечная дата для создания данных.
        :param phenotype:
            Параметры данных, модели, оптимизатора и политики обучения.
        :param pickled_model:
            Сохраненные параметры для натренированной модели.
        """
        self._tickers = tickers
        self._end = end
        self._phenotype = phenotype
        self._pickled_model = pickled_model
        self._model = None
        self._llh = None
        self.sn_comments = sn_comments


    def __bytes__(self) -> bytes:
        """Сохраненные параметры для натренированной модели."""
        if self._pickled_model is not None:
            return self._pickled_model

        if self._model is None:
            return b""

        buffer = io.BytesIO()
        self._model.to("cpu")
        state_dict = self._model.state_dict()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    @property
    def quality_metrics(self) -> tuple[float, float]:
        """Логарифм правдоподобия."""
        if self._llh is None:
            self._llh = self._eval_llh()
        return self._llh

    def prepare_model(self, loader: data_loader.DescribedDataLoader) -> nn.Module:
        """Загрузка или обучение модели."""
        if self._model is not None:
            return self._model

        pickled_model = self._pickled_model
        if pickled_model:
            self._model = self._load_trained_model(pickled_model, loader)
        else:
            self._model = self._train_model()

        return self._model


    def _eval_llh(self) -> tuple[float, float]:
        """Вычисляет логарифм правдоподобия.

        Прогнозы пересчитываются в дневное выражение для сопоставимости и вычисляется логарифм
        правдоподобия. Модель загружается при наличии сохраненных весов или обучается с нуля.
        """
#        LOGGER.info(f"SNEDIT_010: Before DescribedDataLoader")
## Тут основной тормоз во время пилы
## Причем именно при вызове этой подпрограммы, а не внутри нее.   Либо при вызове происходит инициация класса.

#        import pickle
#        collection_speedy = snspeedy_get_collection()
#
##save
#        import bson
#        speedy_id = collection_speedy.insert_one({
#            "bin_var1": bson.Binary(pickle.dumps( self._phenotype["data"] )),
##            "bin_var2": bson.Binary(pickle.dumps(state_dict)),
#        })


        loader = data_loader.DescribedDataLoader(
            self._tickers,
            self._end,
            self._phenotype["data"],
            data_params.TestParams,
#            speedy_id.inserted_Orid,
#            "snParam1",
        )
#        LOGGER.info(f"SNEDIT_011: After DescribedDataLoader")

        n_tickers = len(self._tickers)
        days, rez = divmod(len(loader.dataset), n_tickers)
        if rez:
            history = int(self._phenotype["data"]["history_days"])

            raise TooLongHistoryError(f"Слишком большая длинна истории - {history}. len(loader.dataset)={len(loader.dataset)}, n_tickers={n_tickers}, days={days}, rez={rez}")

#        LOGGER.info(f"SNEDIT_012: After DescribedDataLoader")
## Тут основная долгая загрузка процессора. Вопросов нет
        model = self.prepare_model(loader)
#        LOGGER.info(f"SNEDIT_013: After DescribedDataLoader")
        model.to(DEVICE)
#        LOGGER.info(f"SNEDIT_014: After DescribedDataLoader")
        loss_fn = log_normal_llh_mix

        llh_sum = 0
        weight_sum = 0
        all_means = []
        all_vars = []
        all_labels = []

        llh_adj = np.log(data_params.FORECAST_DAYS) / 2
        with torch.no_grad():
            model.eval()
            bars = tqdm.tqdm(loader, file=sys.stdout, desc="~~> Test")
            for batch in bars:
                loss, mean, var = loss_fn(model, batch)
                llh_sum -= loss.item()
                weight_sum += mean.shape[0]
                all_means.append(mean)
                all_vars.append(var)
                all_labels.append(batch["Label"])

                bars.set_postfix_str(f"{llh_sum / weight_sum + llh_adj:.5f}")

#        LOGGER.info(f"SNEDIT_015: After DescribedDataLoader")
        all_means = torch.cat(all_means).cpu().numpy().flatten()
        all_vars = torch.cat(all_vars).cpu().numpy().flatten()
        all_labels = torch.cat(all_labels).cpu().numpy().flatten()
        llh = llh_sum / weight_sum + llh_adj

#        LOGGER.info(f"SNEDIT_016: After DescribedDataLoader")
# Тут выводит параметры шага пилы.  Можно добавить номер шага, но его придется передавать снаружи
        ir = _opt_port(
            all_means,
            all_vars,
            all_labels,
            self._tickers,
            self._end,
            self._phenotype,
            self.sn_comments,
        )
#        LOGGER.info(f"SNEDIT_017: After DescribedDataLoader")

        return llh, ir

    def _load_trained_model(
        self,
        pickled_model: bytes,
        loader: data_loader.DescribedDataLoader,
    ) -> nn.Module:
        """Создание тренированной модели."""
        model = self._make_untrained_model(loader)
        buffer = io.BytesIO(pickled_model)
        state_dict = torch.load(buffer)
        model.load_state_dict(state_dict)
        return model

    def _make_untrained_model(
        self,
        loader: data_loader.DescribedDataLoader,
    ) -> nn.Module:
        """Создает модель с не обученными весами."""
        model_type = getattr(models, self._phenotype["type"])
        model = model_type(loader.history_days, loader.features_description, **self._phenotype["model"])

        if (n_par := sum(tensor.numel() for tensor in model.parameters())) > MAX_DOC_SIZE:
            raise TooLargeModelError(f"Очень много параметров: {n_par}")

        return model

    def _train_model(self) -> nn.Module:
        """Тренировка модели."""
        phenotype = self._phenotype

        try:
            loader = data_loader.DescribedDataLoader(
                self._tickers,
                self._end,
                phenotype["data"],
                data_params.TrainParams,
#                "snParam2",
            )
        except ValueError:
            history = int(self._phenotype["data"]["history_days"])

            raise TooLongHistoryError(f"Слишком большая длина истории: {history}")

        if len(loader.features_description) == 1:
            raise DegeneratedModelError("Отсутствуют активные признаки в генотипе")

        model = self._make_untrained_model(loader)
        model.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), **phenotype["optimizer"])

        steps_per_epoch = len(loader)
        scheduler_params = dict(phenotype["scheduler"])
        epochs = scheduler_params.pop("epochs")
        total_steps = 1 + int(steps_per_epoch * epochs)
        scheduler_params["total_steps"] = total_steps
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)

        LOGGER.info(f"Epochs - {epochs:.2f} / Train size - {len(loader.dataset)}")
        modules = sum(1 for _ in model.modules())
        model_params = sum(tensor.numel() for tensor in model.parameters())
        LOGGER.info(f"Количество слоев / параметров - {modules} / {model_params}")

        batch_size = (model_params * 4) * self._phenotype["data"]["batch_size"] / (2**10) ** 3
        if batch_size > MAX_BATCH_SIZE:
            raise TooLargeModelError(f"Размер батча {batch_size:.0f} > {MAX_BATCH_SIZE}Gb")

        llh_sum = 0
        llh_deque = collections.deque([0], maxlen=steps_per_epoch)
        weight_sum = 0
        weight_deque = collections.deque([0], maxlen=steps_per_epoch)
        loss_fn = log_normal_llh_mix

        loader = itertools.repeat(loader)
        loader = itertools.chain.from_iterable(loader)
        loader = itertools.islice(loader, total_steps)

        model.train()
        bars = tqdm.tqdm(loader, file=sys.stdout, total=total_steps, desc="~~> Train")
        llh_min = None
        llh_adj = np.log(data_params.FORECAST_DAYS) / 2
        for batch in bars:
            optimizer.zero_grad()

            loss, means, _ = loss_fn(model, batch)

            llh_sum += -loss.item() - llh_deque[0]
            llh_deque.append(-loss.item())

            weight_sum += means.shape[0] - weight_deque[0]
            weight_deque.append(means.shape[0])

            loss.backward()
            optimizer.step()
            scheduler.step()

            llh = llh_sum / weight_sum + llh_adj
            bars.set_postfix_str(f"{llh:.5f}")

            if llh_min is None:
                llh_min = llh - LLH_DRAW_DOWN

            total_time = bars.format_dict
            total_time = total_time["total"] / (1 + total_time["n"]) * total_time["elapsed"]
            if total_time > DAY_IN_SECONDS:
                raise DegeneratedModelError(f"Большое время тренировки: {total_time:.0f} >" f" {DAY_IN_SECONDS}")

            # Такое условие позволяет отсеять NaN
            if not (llh > llh_min):
                raise GradientsError(f"LLH снизилось - начальное: {llh_min + LLH_DRAW_DOWN:0.5f}")

        return model

    def forecast(self) -> Forecast:
        """Прогноз годовой доходности."""
        loader = data_loader.DescribedDataLoader(
            self._tickers,
            self._end,
            self._phenotype["data"],
            data_params.ForecastParams,
        )

        model = self.prepare_model(loader)
        model.to(DEVICE)

        means = []
        stds = []
        with torch.no_grad():
            model.eval()
            for batch in loader:
                dist = model.dist(batch)

                means.append(dist.mean - torch.tensor(1.0))
                stds.append(dist.variance**0.5)

        means = torch.cat(means, dim=0).cpu().numpy().flatten()
        stds = torch.cat(stds, dim=0).cpu().numpy().flatten()

        means = pd.Series(means, index=list(self._tickers))
        means = means.mul(YEAR_IN_TRADING_DAYS / data_params.FORECAST_DAYS)

        stds = pd.Series(stds, index=list(self._tickers))
        stds = stds.mul((YEAR_IN_TRADING_DAYS / data_params.FORECAST_DAYS) ** 0.5)

        return Forecast(
            tickers=self._tickers,
            date=self._end,
            history_days=self._phenotype["data"]["history_days"],
            mean=means,
            std=stds,
            max_std=self._phenotype["utility"]["max_std"],
        )


def _opt_port(
    mean: np.array,
    var: np.array,
    labels: np.array,
    tickers: tuple[str],
    end: pd.Timestamp,
    phenotype: PhenotypeData,
    sn_comment: str = ""
) -> float:
    """Доходность портфеля с максимальными ожидаемыми темпами роста.

    Рассчитывается доходность оптимального по темпам роста портфеля в годовом выражении (RET) и
    выводится дополнительная статистика:

    - MEAN - доходность равновзвешенного портфеля в качестве простого бенчмарка
    - PLAN - ожидавшаяся доходность. Большие по модулю значения потенциально говорят о не адекватности
    модели
    - STD - ожидавшееся СКО. Большие по значения потенциально говорят о не адекватности модели
    - DD - грубая оценка ожидаемой просадки
    - POS - количество не нулевых позиций. Малое количество говорит о слабой диверсификации портфеля
    - MAX - максимальный вес актива. Большое значение говорит о слабой диверсификации портфеля
    """
    mean *= YEAR_IN_TRADING_DAYS / data_params.FORECAST_DAYS
    var *= YEAR_IN_TRADING_DAYS / data_params.FORECAST_DAYS
    labels *= YEAR_IN_TRADING_DAYS / data_params.FORECAST_DAYS

    w, sigma = _opt_weight(mean, var, tickers, end, phenotype)
    ret = (w * labels).sum()
    ave = labels.mean()
    delta = ret - ave
    ret_plan = (w * mean).sum()
    std_plan = (w.reshape(1, -1) @ sigma @ w.reshape(-1, 1)).item() ** 0.5

    l_wgts = []   # список

    c=0
    for t in tickers:
        l_wgts.append([ w[c]*100, t])
        c=c+1
    l_wgts.sort(reverse=True)

    c=0
    max_t=""
    for wg in l_wgts:
        if wg[0] <= 1 or c > 10:
            break
        max_t= max_t + f"{wg[1]}: {wg[0]:.2f}  "
        c=c+1



    LOGGER.info(
        " / ".join(
            [
                f"Delta = {delta:.2%}",
                f"RET = {ret:.2%}",
                f"AVE = {ave:.2%}",
                f"PLAN = {ret_plan:.2%}",
                f"STD = {std_plan:.2%}",
                f"POS = {int(1 / (w ** 2).sum())}",
                f"MAX = {w.max():.2%}",
                f"{sn_comment}",
                f"\n{max_t}",
            ],
        ),
    )



    if sn_comment != "":
        from datetime import datetime
        now = datetime.now()
        curdt=now.strftime("%Y-%m-%d %H:%M:%S")

        f = open("/home/sn/sn/poptimizer-master/stat_POS.txt",'a')

        if f.tell() == 0:
            f.write(f"now\tsn_comment\t\t\t\tDELTA\tRET\tAVE\tRET_PLAN\tSTD_PLAN\tPOS\tMAX\tTICKERS\n")
        else:
            f.write(f"{curdt}\t{sn_comment}\t{delta*100:.2f}\t{ret*100:.2f}\t{ave*100:.2f}\t{ret_plan*100:.2f}\t{std_plan*100:.2f}\t{int(1 / (w ** 2).sum())}\t{w.max()*100:.2f}\t{max_t}\n")
        f.close()


    return delta


def _opt_weight(
    mean: np.array,
    variance: np.array,
    tickers: tuple[str],
    end: pd.Timestamp,
    phenotype: PhenotypeData,
) -> tuple[np.array, np.array]:
    """Веса портфеля с максимальными темпами роста и использовавшаяся ковариационная матрица.

    Задача максимизации темпов роста портфеля сводится к максимизации математического ожидания
    логарифма доходности. Дополнительно накладывается ограничение на полною отсутствие кэша и
    неотрицательные веса отдельных активов.
    """
    history_days = phenotype["data"]["history_days"]
    mean = mean.reshape(-1, 1)

    sigma = ledoit_wolf.ledoit_wolf_cor(tickers, end, history_days, config.FORECAST_DAYS)[0]
    std = variance**0.5
    sigma = std.reshape(1, -1) * sigma * std.reshape(-1, 1)

    w = np.ones_like(mean).flatten()

    rez = optimize.minimize(
        lambda x: -(x.reshape(1, -1) @ mean).item(),
        w,
        jac=lambda x: -mean.flatten(),
        method="SLSQP",
        bounds=[(0, None) for _ in w],
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: x.sum() - 1,
                "jac": lambda x: np.ones_like(x),
            },
            {
                "type": "ineq",
                "fun": lambda x: phenotype["utility"]["max_std"] ** 2
                - (x.reshape(1, -1) @ sigma @ x.reshape(-1, 1)).item(),
                "jac": lambda x: (-2 * sigma @ x.reshape(-1, 1)).flatten(),
            },
        ],
    )

    return rez.x / rez.x.sum(), sigma
