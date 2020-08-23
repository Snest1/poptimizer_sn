"""Интерфейсы взаимодействия."""
import abc
from datetime import datetime
from typing import Iterable, Literal, NamedTuple, Optional, Tuple

import pandas as pd

from poptimizer.config import POptimizerError


class DataError(POptimizerError):
    """Ошибки связанные с операциями по обновлению данных."""


# Наименования групп таблиц
TRADING_DATES = "trading_dates"
CONOMY = "conomy"


class TableName(NamedTuple):
    """Наименование таблицы."""

    group: Literal["trading_dates", "conomy"]
    name: str


class TableTuple(NamedTuple):
    """Представление таблицы в виде кортежа."""

    group: str
    name: str
    df: pd.DataFrame
    timestamp: datetime


class AbstractUpdater(abc.ABC):
    """Обновляет конкретную группу таблиц."""

    @abc.abstractmethod
    def get_update(self, table_name: TableName) -> pd.DataFrame:
        """Загружает обновление."""


class AbstractDBSession(abc.ABC):
    """Сессия работы с базой данных."""

    @abc.abstractmethod
    def get(self, name: Tuple[str, str]) -> Optional[TableTuple]:
        """Получает данные из хранилища."""

    @abc.abstractmethod
    def commit(self, tables_vars: Iterable[TableTuple]) -> None:
        """Сохраняет данные таблиц."""