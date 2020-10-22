"""Информация о актуальности данных по дивидендам."""
from typing import List, Tuple

import numpy as np
import pandas as pd

from poptimizer.data.config import bootstrap
from poptimizer.data.ports import col, outer
from poptimizer.data.views.crop import div


def smart_lab() -> pd.DataFrame:
    """Информация по дивидендам с smart-lab.ru."""
    table_name = outer.TableName(outer.SMART_LAB, outer.SMART_LAB)
    requests_handler = bootstrap.get_handler()
    return requests_handler.get_df(table_name)


def new_on_smart_lab(tickers: Tuple[str, ...]) -> List[str]:
    """Список тикеров с новой информацией о дивидендах на SmartLab.

    Выбираются только тикеры из предоставленного списка.

    :param tickers:
        Тикеры, для которых нужно проверить актуальность данных.
    :return:
        Список новых тикеров.
    """
    status = []
    for ticker, date, div_value in smart_lab().itertuples():
        if ticker not in tickers:
            continue

        df = div.dividends(ticker)
        if date not in df.index:
            status.append(ticker)
        elif not np.isclose(df.loc[date, ticker], div_value):
            status.append(ticker)

    if status:
        print("\nДАННЫЕ ПО ДИВИДЕНДАМ ТРЕБУЮТ ОБНОВЛЕНИЯ\n")  # noqa: WPS421
        print(", ".join(status))  # noqa: WPS421

    return status


def _compare(source_name: str, df_local: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает данные по дивидендам из двух источников."""
    df = pd.concat([df_local, df_source], axis="columns")
    df.columns = ["LOCAL", "SOURCE"]
    df["STATUS"] = "ERROR"
    equal_div = np.isclose(df.iloc[:, 0], df.iloc[:, 1])
    df.loc[equal_div, "STATUS"] = ""

    print(f"\nСРАВНЕНИЕ ЛОКАЛЬНЫХ ДАННЫХ С {source_name}\n\n{df}")  # noqa: WPS421

    return df


def dividends_validation(ticker: str) -> None:
    """Проверяет корректности данных о дивидендах для тикера.

    Сравнивает основные данные по дивидендам с альтернативными источниками и распечатывает результаты
    сравнения.

    :param ticker:
        Тикер.
    """
    df_div = div.dividends(ticker, force_update=True)

    _compare("dohod.ru", df_div, div.dohod(ticker))
    _compare("conomy.ru", df_div, div.conomy(ticker))
    _compare("bcs-express.ru", df_div, div.bcs(ticker))

    df = smart_lab()
    df = df.loc[df.index == ticker]
    df = df.set_index(col.DATE)
    _compare("smart-lab.ru", df_div, df)
