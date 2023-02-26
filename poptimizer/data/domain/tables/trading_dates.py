"""Таблица с торговыми датами."""
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import ClassVar, Final, Optional

import pandas as pd

from poptimizer.data import ports
from poptimizer.data.adapters.gateways import moex
from poptimizer.data.domain import events
from poptimizer.data.domain.tables import base
from poptimizer.shared import domain

# Часовой пояс MOEX
_MOEX_TZ: Final = zoneinfo.ZoneInfo(key="Europe/Moscow")

# Торги заканчиваются в 24.00, но данные публикуются 00.45
#_END_HOUR: Final = 0
#_END_MINUTE: Final = 45
_END_HOUR: Final = 4    # Но мои валюты на трейдингвью закрываются скорее всего около 3.00  На всякий случай +1 час на зимнее время
_END_MINUTE: Final = 5

def _to_utc_naive(date: datetime) -> datetime:
    """Переводит дату в UTC и делает ее наивной."""
    date = date.astimezone(timezone.utc)
    return date.replace(tzinfo=None)


def _trading_day_potential_end() -> datetime:
    """Возможный конец последнего торгового дня UTC."""
    now = datetime.now(_MOEX_TZ)
    end_of_trading = now.replace(
        hour=_END_HOUR,
        minute=_END_MINUTE,
        second=0,
        microsecond=0,
    )

    if end_of_trading > now:
        end_of_trading -= timedelta(days=1)

    return _to_utc_naive(end_of_trading)


class TradingDates(base.AbstractTable[events.DateCheckRequired]):
    """Таблица с данными о торговых днях.

    Обрабатывает событие начала работы приложения.
    Инициирует событие в случае окончания очередного торгового дня.
    """

    group: ClassVar[ports.GroupName] = ports.TRADING_DATES
    _gateway: Final = moex.TradingDatesGateway()

    def __init__(
        self,
        id_: domain.ID,
        df: Optional[pd.DataFrame] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Сохраняет необходимые данные и кэширует старое значение."""
        super().__init__(id_, df, timestamp)
        self._last_trading_day_old: Optional[datetime] = None

    def _update_cond(self, event: events.DateCheckRequired) -> bool:
        """Обновляет, если последняя дата обновления после потенциального окончания торгового дня."""
        if self._timestamp is None:    # Если запись с торговыми днями была удалена, то точно нужно обновление курсов
            return True
        retval = _trading_day_potential_end() > self._timestamp  # snedit  StopUpdate

# SNEDIT  отключение импорта новых котировок
        try:
            f = open("/home/sn/sn/poptimizer-master/auto/!work/!moex_stop",'r')
            f.close()
            return False
        except:
            return _trading_day_potential_end() > self._timestamp

    async def _prepare_df(self, event: events.DateCheckRequired) -> pd.DataFrame:
        """Загружает новый DataFrame."""
        return await self._gateway()

    def _validate_new_df(self, df_new: pd.DataFrame) -> None:
        """Проверка корректности индекса и заголовков - потом сохраняет старые данные."""
        if df_new.index.tolist() != [0]:
            raise base.TableIndexError()
        if df_new.columns.tolist() != ["from", "till"]:
            raise base.TableIndexError()

        if (df := self._df) is not None:
            self._last_trading_day_old = df.loc[0, "till"]

    def _new_events(self, event: events.DateCheckRequired) -> list[domain.AbstractEvent]:
        """Событие окончания торгового дня."""
        df: pd.DataFrame = self._df
        last_trading_day = df.loc[0, "till"]
        if last_trading_day != self._last_trading_day_old:
            return [events.TradingDayEnded(last_trading_day.date())]
        return []
