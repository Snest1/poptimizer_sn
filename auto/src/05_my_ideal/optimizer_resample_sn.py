"""Оптимизатор портфеля на основе ресемплирования отдельных прогнозов."""
import logging

import numpy as np
import pandas as pd
from scipy import stats

from poptimizer import config
from poptimizer.portfolio import metrics
from poptimizer.portfolio.portfolio import CASH, Portfolio

from scipy import stats
from sklearn.preprocessing import quantile_transform


# Наименование столбцов
_PRIORITY = "PRIORITY"
_LOWER = "LOWER"
_UPPER = "UPPER"
_COSTS = "COSTS"
_SELL = "SELL"
_BUY = "BUY"
_PENDING = "PENDING"
_SIGNAL = "SIGNAL"
_WEIGHT = "WEIGHT"
_RISK_CON = "RISK_CON"


class Optimizer:  # noqa: WPS214
    """Предлагает сделки для улучшения метрики портфеля.

    Использует множество предсказаний и статистические тесты для выявления только статистически значимых
    улучшений портфеля, которые покрывают транзакционные издержки и воздействие на рыночные котировки.
    Рекомендации даются в сокращенном виде без конкретизации конкретных сделок.
    """

    def __init__(self, portfolio: Portfolio, *, p_value: float = config.P_VALUE, for_sell: int = 1):
        """Учитывается градиент, его ошибку и ликвидность бумаг.

        :param portfolio:
            Оптимизируемый портфель.
        :param p_value:
            Требуемая значимость отклонения градиента от нуля.
        :param for_sell:
            Количество претендентов на продажу.
        """
        self._portfolio = portfolio
        self._p_value = p_value
        self._metrics = metrics.MetricsResample(portfolio)
        self._for_sell = for_sell - 1
        self._logger = logging.getLogger()


    def _update_portfolio(self, rec, cash):
        """Создаёт новый портфель с учётом новой транзакции"""
        rec["SHARES"] = rec["lots"] * rec["LOT_size"]
        cur_prot = Portfolio(
            name=self.portfolio.name,
            date=self.portfolio.date,
            cash=cash,
            positions=rec["SHARES"].to_dict(),
        )
        return cur_prot

    @property
    def __str__(self) -> str:
        """Информация о позициях, градиенты которых значимо отличны от 0."""

        old_prot = self.portfolio

        import time as tm
        sn_trading_ops = pd.DataFrame(columns=[
            'Ticker',
            'op',
            'Lots_chg',
        ])
#        sn_trading_ops.drop

        def sn_quit(old_prot : Portfolio):
            self._logger.info(cur_prot)
            self._logger.info(self._metrics)

            sn_trading_ops_fine = pd.DataFrame(columns=[
                'Ticker',
                'Mult',
                'Lots_was',
                'Summ_was',
                'op',
                'Lots_chg',
                'Summ_chg',
                'Lots_new',
                'Summ_new',
            ])

            grouped_df = sn_trading_ops.groupby(["Ticker", "op"], as_index=False, axis=0).sum()
            grouped_df.sort_values(['Lots_chg'], ascending=[True], inplace=True)

            for i, r in grouped_df.iterrows():
                sn_row = {
                    'Ticker': r['Ticker'],
                    'Mult': cur_prot.lot_size[r['Ticker']],
                    'Lot_price': cur_prot.price[r['Ticker']],
                    'Lots_was': old_prot.lots[r['Ticker']],
                    'Summ_was': old_prot.value[r['Ticker']],
                    'op': r['op'],
                    'Lots_chg': r['Lots_chg'],
                    'Summ_chg': cur_prot.value[r['Ticker']] - old_prot.value[r['Ticker']],
                    'Lots_new': cur_prot.lots[r['Ticker']],
                    'Summ_new': cur_prot.value[r['Ticker']],
                }
                row2df = pd.DataFrame.from_records([sn_row])
                sn_trading_ops_fine = pd.concat([sn_trading_ops_fine, row2df], ignore_index=True,
                                           axis=0)  # how to handle index depends on context

            sn_trading_ops_fine.sort_values(['Summ_chg'], ascending=[True], inplace=True)
            sn_trading_ops_fine.to_excel('/home/sn/sn/poptimizer-master/my_dumps/ops_' + tm.strftime("%Y%m%d_%H%M%S", tm.gmtime()) + '.xlsx', index=False)

            self._logger.info(sn_trading_ops)
            self._logger.info(sn_trading_ops_fine)

            quit()
            return

        #        self._portfolio = portfolio.positions

        cur_prot = self.portfolio

        file = open("./out.txt", "w")

#        import os
#        directory = '/home/sn/sn/poptimizer-master/auto/!work'
#        files = os.listdir(directory)
#        bashes = filter(lambda x: x.endswith('.sh'), files)

#        for bash in bashes:
#             self._logger.info(f'Running {directory}/{bash}')
#             os.system(f'{directory}/{bash} > {directory}/{bash}_out')
#             os.system(f'rm {directory}/{bash}')
#        quit()



        cur_prot.add_tickers()
        cur_prot.remove_tickers()


### buy and sell
        if 1 == 1:
            buy_vol_mult = {   # Словарь множителей объема.  Если бумаги нет, то = 1
'BTCRUB': 5,  # майню, поэтому нужно докупать.
'ETHRUB': 3,  # могу обменять на намайненное
'GAZP': 2,
'SBER': 2,
'SBERP': 2,
            }

            buy_prior = {   # Словарь приоритетов
'PLZL': 1995,     # как GOLD
'YNDX': 4002,     # GDR
'AGRO': 4003,     # GDR
'FIVE': 4004,     # GDR
'GLTR': 4005,     # GDR
'BTCRUB': 1006,   # майню, поэтому нужно докупать.
'ETHRUB': 1007,   # могу обменять на намайненное

'SNGSP': 1030,
'LSNGP': 1080,
'TTLK': 1090,
'BISVP': 1095,
'GMKN': 1097,
'SBER': 1102,
'SBERP': 1103,
'GAZP': 1105,
'NVTK': 1110,
'RASP': 1130,
'BSPB': 1140,
'MGNT': 1170,
'MOEX': 1180,
'AKRN': 1200,
'PHOR': 1210,
'LKOH': 1305,
'ROSN': 1310,
'SIBN': 1320,
'TATN': 1401,
'TATNP': 1402,
'VSMO': 1502,
'RUAL': 1505,
'CHMF': 1510,
'MAGN': 1520,
'NLMK': 1530,
'CHMK': 1550,
'VTBR': 1600,
'VKCO': 2090,
'ALRS': 2100,
'PIKK': 2210,
'LSRG': 2220,
'ETLN': 2230,
'SMLT': 2240,
'INGR': 2250,
'MSNG': 2310,
'MRKP': 2320,
'RSTI': 2360,
'RSTIP': 2365,
'MGTSP': 2410,
'TRMK': 2500,
'NKNCP': 2700,
'OGKB': 2830,
'NMTP': 2850,
'BANEP': 2890,
'RTKMP': 2900,
'RTKM': 2903,
'MTSS': 2930,
'UPRO': 2950,
'KRKNP': 3080,
'JNOS': 3090,
'JNOSP': 3095,
'SELG': 3097,
'AFKS': 3110,
'PMSBP': 3205,
'DSKY': 3250,
'ENPG': 3280,
'SGZH': 3300,
'FEES': 3310,
'IRAO': 3320,
'FESH': 3350,
'RNFT': 3500,
'HIMCP': 3510,
'ENRU': 3800,
'TGKB': 3910,
'NKHP': 3950,
'PMSB': 3990,
'HHRU': 4100,
'OKEY': 4200,
'MDMG': 4300,
'GCHE': 4400,
'FIXP': 4500,
'TCSG': 5040,
'QIWI': 5050,
'OZON': 5080,
'MVID': 5090,
'BELU': 5115,
'AQUA': 5120,
'ISKJ': 5150,
'FLOT': 5210,
#'CBOM': 5220,
'KMAZ': 5250,
'KAZTP': 5310,
'KAZT': 5320,
'HYDR': 5510,
'TGKA': 5520,
'GTRK': 5530,
'BRZL': 5610,
'MRKV': 5910,
'MSRS': 5950,
'VJGZ': 6110,
'VJGZP': 6120,
'IGST': 6210,
'KZOSP': 6220,
'LIFE': 6500,
'MTLR': 7110,
'MTLRP': 7120,
'KZOS': 7210,
'RGSS': 7210,
'MRSB': 7310,
'KROT': 7410,
'ABRD': 7510,
'AMEZ': 7610,
'POLY': 7810,
'TRNFP': 7910,
'ROLO': 7950,
'LVHK': 8110,
'KRSB': 8210,
'RZSB': 8310,
'RKKE': 8410,
'MRKZ': 8510,
'UNKL': 8610,
'USBN': 8710,
'MGNZ': 8810,
'AFLT': 9110,
'LSNG': 9150,
'TGKN': 9210,
'SLEN': 9250,
'MRKY': 9310,
'MSTT': 9350,
'DVEC': 9410,
'KLSB': 9450,
'MRKC': 9510,
'DIOD': 9550,
'GEMA': 9610,
'NKNC': 9650,
'SVAV': 9710,
'YAKG': 9750,
'BLNG': 9810,
'ZILL': 9850,
'MRKS': 9910,
'NFAZ': 9930,
'MRKU': 9960,
'UNAC': 9970,
'IRKT': 9980,
'SNGS': 9990,
'CNTL': 9995,
'CNTLP': 9997,
'BANE': 9998,


'DIVD': 12100,          # Бежать из акций лучше в фонды, чем в рубли
'EQMX': 12200,          # Бежать из акций лучше в фонды, чем в рубли
'SBMX': 12300,          # Бежать из акций лучше в фонды, чем в рубли
'TMOS': 12400,          # Бежать из акций лучше в фонды, чем в рубли

'GOLD': 1995,          # лучше сейчас золото, чем рубли.    Но похоже, что при увеличении золота, LQDT совсем не уменьшеается...
'GLDRUB_TOM': 1990,          # лучше сейчас золото, чем рубли.    Но похоже, что при увеличении золота, LQDT совсем не уменьшеается...
'SLVRUB_TOM': 1980,          # лучше сейчас золото, чем рубли.    Но похоже, что при увеличении золота, LQDT совсем не уменьшеается...

#'LQDT': 999,          # лучше сейчас золото, чем рубли.    Но похоже, что при увеличении золота, LQDT совсем не уменьшеается...
            }

            zero_buying = [
## 'LQDT',               # Закомментировал, т.к. готов покупать LQDT вместо бумаг
#			# НИЖЕ ИСТОРИЧЕКСИЕ КОММЕНТАРИИ
#			# закомментировал, т.к. покупка кэша означает необходимость продавать бумаги! нельзя его бесплатно купить.
#			# а может он и облигации скажет продавать, а так как я сейчас кэш бесплатно беру, то он и облиги к нему подтягивает
#			# Проверю - не заставит ли продать вообще все, чтобы в кэш выйти
#			# либо быть готовым довнести (тогда надо начальную сумму менять, а не покупать бесплатно)
## 'AKMB', 'OBLG', 'SBGB', 'SBRB',                      # в этом портфеле врядли придется это раскомментировать
## 'GLDRUB_TOM', 'SLVRUB_TOM', 'GOLD',   # Готов покупать/продавать при необходимости.
#'BTCRUB', 'ETHRUB',      # в этом портфеле покупка/продажа крипты только виртуальная, чтобы понять необходимый в портфеле процент крипты. Разбивка этой суммы по крипто-бумагам в другом портфеле
#'PLZL', 'YNDX', 	   # это последние бумаги, котрые после после размышлений 24.09.2022 решил мониторить по своему разумению
#'FIVE',                    # эта зависла на ирином ИИС, и не могу продать, пока не закрою ИИС.  Но буду бесплатно покупать, пока их не наберется больше, чем у меня есть
#'GLTR',                    # эта зависла на моем ИИС, и не могу продать, пока не закрою ИИС. Но буду бесплатно покупать, пока их не наберется больше, чем у меня есть
#'AGRO',                    # эта зависла на моем ИИС, и не могу продать, пока не закрою ИИС. Но буду бесплатно покупать, пока их не наберется больше, чем у меня есть
]

            skip_buying = [
# 'LQDT',                  #    Готов продавать бумаги и держать деньги в кэше
'AKMB', 'OBLG', 'SBGB', 'SBRB',		# 2022.09.24 принято решение не докупать облигации сейчас. Докупать их можно на грандиозном падении за кэш/LQDT.  Но продавать можно
# 'GLDRUB_TOM', 'SLVRUB_TOM', 'GOLD',   # в этом портфеле врядли придется это раскомментировать
# 'BTCRUB', 'ETHRUB',      # в этом портфеле врядли придется это раскомментировать

'AKME', #	БПИФ	БПИФ «Альфа-Капитал Управляемые российские акции»	0	Исключать			ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ. НЕЛИКВИД
#'DIVD', #	БПИФ	БПИФ «ДОХОДЪ Индекс дивидендных акций РФ»	2428,2	Исключать				ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ.
#'EQMX', #	БПИФ	ВТБ – Индекс МосБиржи	34587	Исключать							ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ.
'RCMX', #	БПИФ	БПИФ «Райффайзен – Индекс МосБиржи полной доходности 15»	2849,4	Исключать		ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ. НЕЛИКВИД
#'SBMX', #	БПИФ	БПИФ «Индекс МосБиржи полной доходности «брутто»» 	17673,8	Исключать			ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ.
'SBRI', #	БПИФ	БПИФ «Сбер – Ответственные инвестиции»	0	Исключать					ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ. НЕЛИКВИД
#'TMOS', #	БПИФ	БПИФ «Тинькофф Индекс Мосбиржи»	60830,42	Исключать					ВРЕМЕННО В КАЧЕСТВЕ ЗАМЕНЫ БУМАГ НА ДНЕ.
'TRUR', #	БПИФ	БПИФ «Тинькофф – Стратегия вечного портфеля в рублях»	676479,44	Исключать
'KOGK', #	Говно	Коршуновский ГОК	36800	Исключать	Низкая ликвидность	Дочка Мечела без дивов
'KUZB', #	Говно		882,5	Исключать	Низкая ликвидность
'LNZL', #	Говно	ЛенЗолото	0	Исключать	Делистинг
'LNZLP', #	Говно	ЛенЗолото	0	Исключать	Делистинг
'NSVZ', #	Говно	Наука и связь	1400	Исключать	Низкая ликвидность
'RBCM', #	Говно		856,2	Исключать	Параша на графике, низкая ликвидность
'RUGR', #	Говно	РусГрэйн	4880	Исключать	Риск банкротства, делистинга
#'SFIN', #	Говно	SFI Инв.холдинг	13968	Исключать	Параша на графике	Инв.холдинг
'TGKD', #	Говно	Квадра	1285	Исключать	Низкая ликвидность, хаи
'VLHZ', #	Говно	Владимирский ХЗ	731,5	Исключать	Параша на графике, низкая ликвидность
'APTK',
'VRSB',		#Низкая ликвидность
'STSBP',	#Низкая ликвидность
'JNOS',		#Низкая ликвидность
'JNOSP',	#Низкая ликвидность
'VJGZ',		#Низкая ликвидность
'VJGZP',	#Низкая ликвидность
'MRKS',		#Низкая ликвидность
'MRSB',		#Низкая ликвидность
'KUZB',		#Низкая ликвидность
'DSKY',		#Выкуп, делистинг
]

            skip_selling = [
# 'LQDT',                  #    раскомментить, когда кончится ликвидный кэш
# 'AKMB', 'OBLG', 'SBGB', 'SBRB',		# 2022.09.24 принято решение не докупать облигации сейчас. Докупать их можно на грандиозном падении за кэш/LQDT.  Но продавать можно
# 'GLDRUB_TOM', 'SLVRUB_TOM', 'GOLD',   # Готов покупать/продавать при необходимости.
# 'BTCRUB', 'ETHRUB',      # в этом портфеле врядли придется это раскомментировать
#'FIVE',                    # эта зависла на ирином ИИС, и не могу продать, пока не закрою ИИС.  Но ей место не тут, где она мешает выровнять портфель, а в zero_selling
#'GLTR',                    # эта зависла на моем ИИС, и не могу продать, пока не закрою ИИС.
]
            zero_selling = [
# 'LQDT',                  # тут никогда не может быть LQDT
# 'AKMB', 'OBLG', 'SBGB', 'SBRB',                      # в этом портфеле врядли придется это раскомментировать
# 'GLDRUB_TOM', 'SLVRUB_TOM', 'GOLD',   # в этом портфеле врядли придется это раскомментировать
#'BTCRUB', 'ETHRUB',      # в этом портфеле покупка/продажа крипты только виртуальная, чтобы понять необходимый в портфеле процент крипты. Разбивка этой суммы по крипто-бумагам в другом портфеле
'PLZL', 'YNDX', 	   # это последние бумаги, котрые после после размышлений 24.09.2022 решил мониторить по своему разумению
'FIVE',                    # эта зависла на ирином ИИС, и не могу продать, пока не закрою ИИС.
'GLTR',                    # эта зависла на моем ИИС, и не могу продать, пока не закрою ИИС.
'AGRO',                    # эта зависла на моем ИИС, и не могу продать, пока не закрою ИИС.

] #

            start_cash = 1000
#            start_cash = 1000
#            deal_size = 1000               # В нормальных условиях ставить 1000 или меньше.
            deal_size = 3000		    # Но пока переходный период у моделей пусть будет 3000.
#            deal_size = 300
            allow_sell = True
            daily_limit = 100000
            allow_daily_relimit = True


### Для тестироавния
#            start_cash = 10000
#            daily_limit = 20000
#            allow_daily_relimit = False


            cur_cash = start_cash
            cur_daily_limit = 0
            daily_limits = 0
            hop_buy = 0
            hop_sell = 0


            self._logger.info(f'OPER\thop_b\thop_s\tcur_daily_limit\tdaily_limits\tbuy\tb_PRIORITY\tlots\tlots_buy\tsumm_b\tsell\ts_PRIORITY\tlots\tlots_sell\tsumm_s\tcash_before')

            algo = 'priority_and_volume'   #simple_priority

# main loop.
            while 1 == 1:

                try:
                    df = self._for_trade()
                except:
                    self._logger.info(f'ERR in _for_trade')
                    #self._logger.info(cur_prot)
                    #self._logger.info(self._metrics)
                    #quit()  ## nothing to buy
                    sn_quit(old_prot)

                rec = None

                cur_metrics = metrics.MetricsResample(cur_prot)
                grads = cur_metrics.all_gradients.iloc[:-2]   # без последних двух строк (CASH и PORTFOLIO) таблицы, в которой строки - бумаги из портфеля, столбцы - градиенты по каждой модели.
            # гармоническое среднее квантилей градиентов вместо бутстрапа
            # вычислительно существенно быстрее
                q_trans_grads = quantile_transform(grads, n_quantiles=grads.shape[0]) # grads.shape[0] - кол-во строк.
            # обработка (маскировка) возможных NA от плохих моделей
                q_trans_grads = np.ma.array(q_trans_grads, mask=~(q_trans_grads > 0))
            # гармоническое среднее сильнее штрафует за низкие значения (близкие к 0),
            # но его использование не принципиально - можно заменить на просто среднее или медиану
                hmean_q_trans_grads = stats.hmean(q_trans_grads, axis=1)
            # учёт оборота при ранжировании
                turnover = quantile_transform(cur_prot.turnover_factor.loc[grads.index].values.reshape(-1, 1), n_quantiles=grads.shape[0])
                priority = stats.hmean(np.hstack([hmean_q_trans_grads.reshape(-1, 1), turnover]), axis=1)
                rec = pd.Series(data=priority, index=grads.index).to_frame(name="PRIORITY")
            #  PRIORITY (from hmean)  не использую, но по другому не умею создать rec  (SNEDIT)

            # так как все операции производятся в лотах, нужно знать стоимость лота и текущее количество лотов
                rec["LOT_size"] = cur_prot.lot_size.loc[rec.index]
                rec["lots"] = (cur_prot.shares.loc[rec.index] / rec["LOT_size"]).fillna(0).astype(int)
                rec["LOT_price"] = (cur_prot.lot_size.loc[rec.index] * cur_prot.price.loc[rec.index]).round(2)
                rec["SHARES"] = rec["lots"] * rec["LOT_size"]

                rec["MEAN"] = cur_metrics.mean.loc[rec.index]
                rec["STD"] = cur_metrics.std.loc[rec.index]
                rec["BETA"] = cur_metrics.beta.loc[rec.index]
                rec["GRAD"] = cur_metrics.gradient.loc[rec.index]

                rec["VALUE"] = cur_prot.value.loc[rec.index].round(2)


                rec['WEIGHT'] = pd.Series(data=df['WEIGHT'], index=df.index)
                rec['RISK_CON'] = pd.Series(data=df['RISK_CON'], index=df.index)
                rec['LOWER'] = pd.Series(data=df['LOWER'], index=df.index)
                rec['UPPER'] = pd.Series(data=df['UPPER'], index=df.index)
                rec['COSTS'] = pd.Series(data=df['COSTS'], index=df.index)
                rec['PRIORITY_2'] = pd.Series(data=df['PRIORITY'], index=df.index)
                rec['SIGNAL'] = pd.Series(data=df['SIGNAL'], index=df.index)


                rec.sort_values(["PRIORITY_2"], ascending=[False], inplace=True)

                file.write(f"hops\tOPER\tTicker\t\
PRIORITY\t\
PRIORITY_2\t\
SIGNAL\t\
lots\t\
WEIGHT\t\
LOT_size\t\
LOT_price\t\
SHARES\t\
VALUE\t\
MEAN\t\
STD\t\
BETA\t\
GRAD\t\
RISK_CON\t\
LOWER\t\
UPPER\t\
COSTS\t\
\n")

                for index, row in rec.iterrows():
                    file.write(f"{hop_buy}\t\
oper\t\
{index}\t\
{row['PRIORITY']:.5f}\t\
{row['PRIORITY_2']:.5f}\t\
{row['SIGNAL']}\t\
{row['lots']}\t\
{row['WEIGHT']:.3f}\t\
{row['LOT_size']}\t\
{row['LOT_price']:.2f}\t\
{row['SHARES']}\t\
{row['VALUE']:.2f}\t\
{row['MEAN']:.5f}\t\
{row['STD']:.5f}\t\
{row['BETA']:.5f}\t\
{row['GRAD']:.5f}\t\
{row['RISK_CON']:.5f}\t\
{row['LOWER']:.5f}\t\
{row['UPPER']:.5f}\t\
{row['COSTS']:.5f}\t\
\n")

#                rec.sort_values(["PRIORITY_2"], ascending=[False], inplace=True)

# select buying ticker with skipping
                skipping = 0
                buy_index = ''

# Подбор из имеющихся вариантов по таблице приоритетов. Начало
                selected_buy = []  #список
                cnt = 0
##                top_prior2 = 0

                while rec.SIGNAL[cnt] == 'BUY':     # Выберем бумаги к покупке кроме skip_buying
                    if rec.PRIORITY_2[cnt].round(5) == 0 and selected_buy == []:
                        self._logger.info(f'Nothing to buy-2. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f}')
                        # self._logger.info(cur_prot)
                        # self._logger.info(self._metrics)
                        # quit()  ## nothing to buy
                        sn_quit(old_prot)

#                    self._logger.info(f'tst1 {rec.index[cnt]} {rec.SIGNAL[cnt]}')
                    if rec.SIGNAL[cnt] == 'BUY' and rec.index[cnt] not in skip_buying:
#                        self._logger.info(f'tst2 {rec.index[cnt]} {rec.SIGNAL[cnt]}')
                        selected_buy.append(rec.index[cnt])
                        if buy_index == '':
                            buy_index = rec.index[cnt]
##                        if top_prior2 == 0:
##                            top_prior2 = rec.PRIORITY_2.loc[cnt]
                    cnt = cnt + 1

                #######

                if (algo == 'simple_priority'):   # Алгоритм основываеся просто на приоритетах
                    cur_buy_priority = 999999     # Выбр по таблице приоритетов к покупке.
                    for sb in selected_buy:
                        prior = buy_prior.get(sb, 0)
                        if prior > 0 and cur_buy_priority > prior:
                            self._logger.info(f'Priority algo: {buy_index}(P2={rec.PRIORITY_2.loc[buy_index]:.5f}, W={rec.WEIGHT.loc[buy_index]:.5f})={cur_buy_priority} replace on {sb}(P2={rec.PRIORITY_2.loc[sb]:.5f}, W={rec.WEIGHT.loc[sb]:.5f})={prior}')
                            cur_buy_priority = prior
                            buy_index = sb
                    if cur_buy_priority != 999999:
                        self._logger.info(f'Priority buy {buy_index} = {cur_buy_priority}')
# Подбор из имеющихся вариантов по таблице приоритетов. Конец.  buy_index - что же решили покупать.

                #######

                if (algo == 'priority_and_volume'):   # Алгоритм учитывает имеющийся объем бумаг, рекомендуемых стандартом к покупке.
#В первую очередь покупать те, которые советует оптимизитор (и которые при этом не в исключениях),  которых нет в портфеле
#Чем больше у меня акций, тем меньше приоритет к покупке
#Если кол-во акций одинаковое - то приоритет бОльшему prior_2
#Если какую-то бумагу я не люблю - то могу понимажть ей prior_2, вполлоть до 0

#Коэф приоритета покупки рассчитывать так
#  - Чем большая сумма  вложена в конкретной бумаге, тем меньше мой коэф приоритета к покупке
#  - Если кол-во акций одинаковое - то приоритет бОльшему prior_2
#  - Свое личное отношение к какой-либо бумаге выражать через множитель (> или < 0) к prior_2, вплоть до 0


#                    selected_buy - список отобранных к покупке бумаг
#                    rec.PRIORITY_2 - приоритеты, определенные стандартом
#                    rec.VALUE - Текущий объем в деньгах
#                    buy_prior - список бумаг с моими приоритетами
#                    buy_vol_mult - список мультипликаторов объема

                    dic = {}
                    for sb in selected_buy:
                        b_prior = buy_prior.get(sb, 0)
                        vol_mult = buy_vol_mult.get(sb, 1)  # Если найден, то = 1
                        prior2 = rec.PRIORITY_2.loc[sb]
                        vol_cur = rec.VALUE.loc[sb]
#                     v1                        v2               v4        v3
# ( ( vol_cur/1000 * b_prior/1000 / vol_mult )  /  prior2  +   1 / (prior2 * vol_mult)  )
                        v1 = vol_cur/1000 * b_prior/1000 / vol_mult
                        v2 = v1 / prior2
                        v3 = prior2 * vol_mult   # Эта и след нужны для сравнения между собой новых бумаг
                        v4 = 1 / v3
                        v_final = v2 + v4   # Чем меньше - тем приоритетнее к покупке.
                        dic[sb] = [ prior2, b_prior, vol_mult, vol_cur, v1, v2, v3, v4, v_final]

                    if len(dic) > 0:
                        df_tmp = pd.DataFrame.from_dict(dic, orient='index', columns = ['prior2', 'buy_prior', 'vol_mult', 'vol_cur', 'v1', 'v2', 'v3', 'v4', 'v_final'])
                        df_tmp = df_tmp.sort_values('v_final')
                        buy_index = df_tmp.index[0]
                        self._logger.info(f'Priority VOL algo selected : {buy_index}')
                        self._logger.info(f'\n{df_tmp}')
                    else:
                        self._logger.info(f'Nothing to buy (Priority VOL algo finished NULL). Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f}')
                        # self._logger.info(cur_prot)
                        # self._logger.info(self._metrics)
                        # quit()  ## nothing to buy
                        sn_quit(old_prot)

                # Подбор из имеющихся вариантов по таблице приоритетов c учетом имеющихся бумаг. Конец.  buy_index - что же решили покупать.
#######


                while buy_index == '':  # Если по приоритетам ничего не нашли, то выберем первый в списке
                    if rec.SIGNAL[skipping] == 'BUY' and rec.index[skipping] not in skip_buying:
                        buy_index = rec.index[skipping]
                    elif rec.index[skipping] in skip_buying:
                        skipping = skipping + 1
                    else:
                        self._logger.info(f'Nothing to buy. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f}')
                        # self._logger.info(cur_prot)
                        # self._logger.info(self._metrics)
                        # quit()  ## nothing to buy
                        sn_quit(old_prot)

                tmp_lots = deal_size // rec.LOT_price.loc[buy_index]

                if tmp_lots > 0:
                    lots_b = tmp_lots
                elif rec.LOT_price.loc[buy_index] < 1:
                    lots_b = 300
                elif rec.LOT_price.loc[buy_index] < 5:
                    lots_b = 100
                elif rec.LOT_price.loc[buy_index] < 10:
                    lots_b = 50
                elif rec.LOT_price.loc[buy_index] < 50:
                    lots_b = 10
                else:
                    lots_b = 1

                summ_b = lots_b * rec.LOT_price.loc[buy_index]


                if summ_b <= cur_cash or buy_index in zero_buying:
                    hop_buy = hop_buy + 1
                    hop_sell = 0
                    if buy_index not in zero_buying:
                        cur_daily_limit = cur_daily_limit + summ_b
                    self._logger.info(f'buy\t{hop_buy}\t{hop_sell}\t{daily_limits}\t{cur_daily_limit:.2f}\t{buy_index}\t{rec.PRIORITY_2.loc[buy_index]:.5f}\t{rec.lots.loc[buy_index]}\t{lots_b:.0f}\t{summ_b:.2f}\t\t\t\t\t\t{cur_cash:.2f}')
                    rec.loc[buy_index, 'lots'] = rec.lots.loc[buy_index] + lots_b
                    sn_row = {
'Ticker': buy_index,
'op': 'BUY',
'Lots_chg': lots_b,
}
                    row2df = pd.DataFrame.from_records([sn_row])
                    sn_trading_ops = pd.concat([sn_trading_ops, row2df], ignore_index=True, axis=0)  # how to handle index depends on context

                    if buy_index not in zero_buying:
                        cur_cash = cur_cash - summ_b
                    else:
                        self._logger.info(f'Zero buy {buy_index} {lots_b:.0f} lot(s) on {summ_b:.2f} (virtual). Now {rec.lots.loc[buy_index]} lot(s).')
                    hop_sell = 0
                    while cur_daily_limit >= daily_limit:
                        daily_limits = daily_limits + 1
                        self._logger.info(f'Daily limit reached. Cash = {cur_cash:.2f}  Daily_limit = {daily_limit:.2f} * {daily_limits}')
                        if daily_limits > 0 and allow_daily_relimit != True:
                            # self._logger.info(cur_prot)
                            # self._logger.info(self._metrics)
                            # quit()  ## nothing to buy
                            sn_quit(old_prot)
                        cur_daily_limit = cur_daily_limit - daily_limit


                elif allow_sell != True:     #no cash
                    self._logger.info(f'Cash is over. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f} * {daily_limits}')
                    #self._logger.info(cur_prot)
                    #self._logger.info(self._metrics)
                    #quit()  ## nothing to buy
                    sn_quit(old_prot)
                else:     #no cash, trying to sell

                    rec.sort_values(["PRIORITY_2"], ascending=[True], inplace=True)
                    skipping = 0
                    sell_index = ''
                    while sell_index == '':
                        if rec.SIGNAL[skipping] == 'SELL' and rec.index[skipping] not in skip_selling:
                            sell_index = rec.index[skipping]
                        elif rec.index[skipping] in skip_selling:
                            skipping = skipping + 1
                        else:
                            self._logger.info(f'Nothing to sell. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f} * {daily_limits}')
                            # self._logger.info(cur_prot)
                            # self._logger.info(self._metrics)
                            # quit()  ## nothing to buy
                            sn_quit(old_prot)

                    hop_sell = hop_sell + 1


                    tmp_lots = deal_size // rec.LOT_price.loc[sell_index]

                    if tmp_lots > 0 and rec.lots.loc[sell_index] > tmp_lots:
                        lots_s = tmp_lots
                    elif rec.LOT_price.loc[sell_index] < 1 and rec.lots.loc[sell_index] > 300:
                        lots_s = 300
                    elif rec.LOT_price.loc[sell_index] < 5 and rec.lots.loc[sell_index] > 100:
                        lots_s = 100
                    elif rec.LOT_price.loc[sell_index] < 10 and rec.lots.loc[sell_index] > 50:
                        lots_s = 50
                    elif rec.LOT_price.loc[sell_index] < 50 and rec.lots.loc[sell_index] > 10:
                        lots_s = 10
                    else:
                        lots_s = 1

                    summ_s = lots_s * rec.LOT_price.loc[sell_index]


                    self._logger.info(f'sell\t{hop_buy}\t{hop_sell}\t{daily_limits}\t{cur_daily_limit:.2f}\t{buy_index}\t{rec.PRIORITY_2.loc[buy_index]:.5f}\t{rec.lots.loc[buy_index]}\t{lots_b:.0f}\t{summ_b:.2f}\t{sell_index}\t{rec.PRIORITY_2.loc[sell_index]:.5f}\t{rec.lots.loc[sell_index]}\t{lots_s:.0f}\t{summ_s:.2f}\t{cur_cash:.2f}')
                    rec.loc[sell_index, 'lots'] = rec.lots.loc[sell_index] - lots_s
                    sn_row = {
'Ticker': sell_index,
'op': 'SELL',
'Lots_chg': lots_s * -1,
}
                    row2df = pd.DataFrame.from_records([sn_row])
                    sn_trading_ops = pd.concat([sn_trading_ops, row2df], ignore_index=True,
                                               axis=0)  # how to handle index depends on context

                    if sell_index not in zero_selling:
                        cur_cash = cur_cash + summ_s
                    else:
                        self._logger.info(f'Zero sell {sell_index} {lots_s:.0f} lot(s) on {summ_s:.2f} (virtual). Now {rec.lots.loc[sell_index]} lot(s).')


                cur_prot = self._update_portfolio(rec, 1)
                self._portfolio = cur_prot
                self._metrics = metrics.MetricsResample(self._portfolio)


            self._logger.info(f'Dummy quit1')
            # self._logger.info(cur_prot)
            # self._logger.info(self._metrics)
            # quit()  ## nothing to buy
            sn_quit()
        self._logger.info(f'Dummy quit2')
        sn_quit()

        #########################################


        hops = 0
        while hops < 20000:
#            self._logger.info(f'SNLOG_HOP: {hops}')
            df = self._for_trade()


            rec = None

            cur_metrics = metrics.MetricsResample(cur_prot)
            grads = cur_metrics.all_gradients.iloc[:-2]
            # гармоническое среднее квантилей градиентов вместо бутстрапа
            # вычислительно существенно быстрее
            q_trans_grads = quantile_transform(grads, n_quantiles=grads.shape[0])
            # обработка (маскировка) возможных NA от плохих моделей
            q_trans_grads = np.ma.array(q_trans_grads, mask=~(q_trans_grads > 0))
            # гармоническое среднее сильнее штрафует за низкие значения (близкие к 0),
            # но его использование не принципиально - можно заменить на просто среднее или медиану
            hmean_q_trans_grads = stats.hmean(q_trans_grads, axis=1)
            # учёт оборота при ранжировании
            turnover = quantile_transform(cur_prot.turnover_factor.loc[grads.index].values.reshape(-1, 1), n_quantiles=grads.shape[0])
            priority = stats.hmean(np.hstack([hmean_q_trans_grads.reshape(-1, 1), turnover]), axis=1)
            rec = pd.Series(data=priority, index=grads.index).to_frame(name="PRIORITY")
            #  PRIORITY (from hmean)  not used, but i cant create  rec  other

#            rec.sort_values(["PRIORITY"], ascending=[False], inplace=True)


            # так как все операции производятся в лотах, нужно знать стоимость лота и текущее количество лотов
            rec["LOT_size"] = cur_prot.lot_size.loc[rec.index]
            rec["lots"] = (cur_prot.shares.loc[rec.index] / rec["LOT_size"]).fillna(0).astype(int)
            rec["LOT_price"] = (cur_prot.lot_size.loc[rec.index] * cur_prot.price.loc[rec.index]).round(2)
            rec["SHARES"] = rec["lots"] * rec["LOT_size"]

            rec["MEAN"] = cur_metrics.mean.loc[rec.index]
            rec["STD"] = cur_metrics.std.loc[rec.index]
            rec["BETA"] = cur_metrics.beta.loc[rec.index]
            rec["GRAD"] = cur_metrics.gradient.loc[rec.index]

            rec["VALUE"] = cur_prot.value.loc[rec.index].round(2)


            rec['WEIGHT'] = pd.Series(data=df['WEIGHT'], index=df.index)
            rec['RISK_CON'] = pd.Series(data=df['RISK_CON'], index=df.index)
            rec['LOWER'] = pd.Series(data=df['LOWER'], index=df.index)
            rec['UPPER'] = pd.Series(data=df['UPPER'], index=df.index)
            rec['COSTS'] = pd.Series(data=df['COSTS'], index=df.index)
            rec['PRIORITY_2'] = pd.Series(data=df['PRIORITY'], index=df.index)
            rec['SIGNAL'] = pd.Series(data=df['SIGNAL'], index=df.index)



            rec.sort_values(["PRIORITY_2"], ascending=[False], inplace=True)
#            rec = rec.iloc[rec.isnull().sum(1).sort_values(["PRIORITY_2"], ascending=[False], inplace=True).index]


            file.write(f"hops\tOPER\tTicker\t\
PRIORITY\t\
PRIORITY_2\t\
SIGNAL\t\
lots\t\
WEIGHT\t\
LOT_size\t\
LOT_price\t\
SHARES\t\
VALUE\t\
MEAN\t\
STD\t\
BETA\t\
GRAD\t\
RISK_CON\t\
LOWER\t\
UPPER\t\
COSTS\t\
\n")

#            file.write(f"{rec}\n")
            for index, row in rec.iterrows():
                file.write(f"{hops}\t\
oper\t\
{index}\t\
{row['PRIORITY']:.5f}\t\
{row['PRIORITY_2']:.5f}\t\
{row['SIGNAL']}\t\
{row['lots']}\t\
{row['WEIGHT']:.3f}\t\
{row['LOT_size']}\t\
{row['LOT_price']:.2f}\t\
{row['SHARES']}\t\
{row['VALUE']:.2f}\t\
{row['MEAN']:.5f}\t\
{row['STD']:.5f}\t\
{row['BETA']:.5f}\t\
{row['GRAD']:.5f}\t\
{row['RISK_CON']:.5f}\t\
{row['LOWER']:.5f}\t\
{row['UPPER']:.5f}\t\
{row['COSTS']:.5f}\t\
\n")



#            file.write(f"{df}\n")


#            import traceback
#            for line in traceback.format_stack():
#                file.write(line.strip())




            # sample port change 
#            rec.loc['GAZP', 'lots'] = rec.lots.loc['GAZP'] + 10

### selling
            if 1 == 2:

#                rec.sort_values(["PRIORITY_2"], ascending=[False], inplace=True)

                sell_index = rec['PRIORITY_2'].idxmin()
                self._logger.info(f'{hops}\tsell_index\t{sell_index}\twas\t{rec.lots.loc[sell_index]}\tpriority_2 =\t{rec.PRIORITY_2.loc[sell_index]}')

#                if rec.LOT_price.loc[sell_index] < 50 and rec.lots.loc[sell_index] > 10:
#                    mult = 10
#                else:
#                    mult = 1

                if rec.LOT_price.loc[sell_index] < 1 and rec.lots.loc[sell_index] > 300:
                    mult = 300
                elif rec.LOT_price.loc[sell_index] < 5 and rec.lots.loc[sell_index] > 100:
                    mult = 100
                elif rec.LOT_price.loc[sell_index] < 10 and rec.lots.loc[sell_index] > 50:
                    mult = 50
                elif rec.LOT_price.loc[sell_index] < 50 and rec.lots.loc[sell_index] > 10:
                    mult = 10
                else:
                    mult = 1
#                mult = 1



#              	rec.loc[::-1, 'lots'] = rec.lots.loc[::-1] - 1
                rec.loc[sell_index, 'lots'] = rec.lots.loc[sell_index] - 1 * mult


### buying
            if 1 == 2:

                rec.sort_values(["PRIORITY_2"], ascending=[False], inplace=True)
#                buy_index = rec['PRIORITY_2'].idxmax()
                skipping = 0
                buy_index = rec.index[skipping]
                while buy_index in ['LQDT', 'OBLG']:
                    skipping = skipping + 1
                    buy_index = rec.index[skipping]


#                self._logger.info(f'buy_index\t{buy_index}')


                self._logger.info(f'{hops}\tbuy_index\t{buy_index}\twas\t{rec.lots.loc[buy_index]}\tpriority_2 =\t{rec.PRIORITY_2.loc[buy_index]}')

                if rec.LOT_price.loc[buy_index] < 1:
                    mult = 300
                elif rec.LOT_price.loc[buy_index] < 5:
                    mult = 100
                elif rec.LOT_price.loc[buy_index] < 10:
                    mult = 50
                elif rec.LOT_price.loc[buy_index] < 50:
                    mult = 10
                else:
                    mult = 1
#                mult = 1

#              	rec.loc[::-1, 'lots'] = rec.lots.loc[::-1] - 1
                rec.loc[buy_index, 'lots'] = rec.lots.loc[buy_index] + 1 * mult



        file.close()

        blocks = [
            "\nSN_OPT",
            f"\nrec = {rec}",
#           f"new_prot = {new_prot}",
        ]

        exit()
        return "\n".join(blocks)



##################################

#        df = self._for_trade()
#        forecasts = self.metrics.count
#        blocks = [
#            "\nОПТИМИЗАЦИЯ ПОРТФЕЛЯ",
#            f"\nforecasts = {forecasts}",
#            f"p-value = {self._p_value:.2%}",
#            f"trading interval = {config.TRADING_INTERVAL}",
#            f"\n{df}",
#        ]
#        return "\n".join(blocks)

    @property
    def portfolio(self) -> Portfolio:
        """Оптимизируемый портфель."""
        return self._portfolio

    @property
    def metrics(self) -> metrics.MetricsResample:
        """Метрики портфеля."""
        return self._metrics

    def _for_trade(self) -> pd.DataFrame:
        """Осуществляет расчет доверительного интервала для среднего."""
        conf_int = self._prepare_bounds()

        break_even = self._break_even(conf_int)

        sell = self._select_sell(conf_int, break_even)

        bye = self._select_buy(break_even, conf_int)


##        self._logger.info(f'SNLOG10: {bye}')
##        bye = bye[bye.index.str.len() >= 5]

##       Говно
#        bye = bye.drop(['FXKZ', 'LNZL', 'LNZLP', 'RKKE', 'SBRI', 'VTBX', 'TRUR', 'VLHZ', 'DIVD', 'LIFE', 'GTRK', 'MSTT', 'RCMX', 'TMOS', 'SBMX', 'TGKN', 'SFIN', 'AKME', 'KROT', 'NSVZ', 'SLEN', 'QIWI', 'LSNG', 'KOGK', 'BANE', 'TGKD', 'RGSS', 'RBCM', 'RUGR', 'RTKM'], errors='ignore')


##      БПИФ
#        bye = bye.drop(['SUGB', 'AKMB', 'SBGB', 'VTBB', 'SBRI', 'VTBX', 'TRUR', 'DIVD', 'RCMX', 'TMOS', 'SBMX', 'AKME', 'EQMX'], errors='ignore')

##      Obligi
#        bye = bye.drop(['BCSB', 'SBRB'], errors='ignore')

##       USA
##        bye = bye[~bye.index.str.contains('-RM', regex=False)]

##       Временно!!!   Отавляет только то, что хочу купить или интересно.  Не включено то, за чем надо наблюдать и остальные бумаги, которые не вошли в первоначальную выборку и не были оценены
##        bye = bye[bye.index.str.contains('FXGD|KAZTP|NKHP|KMAZ|AMEZ|KAZT|AQUA|HIMCP|GCHE|MGNZ|VRSB|KZOSP|TRNFP|PMSB|SELG|ROLO|YAKG|UNAC|SVAV|INGR|RZSB|BRZL|AKRN|NFAZ|USBN|MRKY|SGZH|STSBP|MTLRP|CHMK|ISKJ|BSPB|RUAL|DIOD|NKNC|VTBM|MSNG|SMLT|KZOS|MRKZ|MRSB|GEMA|ZILL|MRKU|RASP|ENPG|MRKP|BELU|TRMK|TGKA|MRKC|MTLR|FESH|MGNT', case=False, regex=True)]



        rez = pd.concat([bye, sell], axis=0)
        rez = rez.sort_values(_PRIORITY, ascending=False)
        rez[_PRIORITY] = rez[_PRIORITY] - break_even

        if len(rez) == 1:
            rez[_SIGNAL] = _PENDING

        return rez

    def _break_even(self, conf_int):
        lower = (conf_int[_LOWER] - conf_int[_COSTS]).max()

        non_zero_positions = self._portfolio.shares.iloc[:-2] > 0
        upper = conf_int[_UPPER].loc[non_zero_positions].sort_values()[self._for_sell]

        return min(lower, upper)

    def _select_buy(self, break_even, conf_int):
        buy = conf_int[_PRIORITY] >= break_even  # noqa: WPS465
        buy = conf_int[buy]
        kwarg = {_SIGNAL: lambda df: _BUY}

        return buy.assign(**kwarg)

    def _select_sell(self, conf_int, break_even):
        sell = conf_int[_UPPER] <= break_even
        sell = sell & (self._portfolio.shares.iloc[:-2] > 0)  # noqa: WPS465
        sell = conf_int[sell]
        kwarg = {
            _PRIORITY: lambda df: df[_UPPER],
            _SIGNAL: lambda df: _SELL,
        }

        return sell.assign(**kwarg)

    def _prepare_bounds(self):
        p_value = self._p_value / (len(self._portfolio.index) - 2) * 2
        conf_int = self.metrics.all_gradients.iloc[:-2]
        conf_int = conf_int.apply(
            lambda grad: _grad_conf_int(grad, p_value),
            axis=1,
            result_type="expand",
        )

        risk_contribution = self._metrics.beta[:-2]
        risk_contribution = risk_contribution * self._portfolio.weight.iloc[:-2]

        conf_int = pd.concat(
            [
                self._portfolio.weight.iloc[:-2],
                risk_contribution,
                conf_int,
            ],
            axis=1,
        )
        conf_int.columns = [_WEIGHT, _RISK_CON, _LOWER, _UPPER]
        conf_int[_COSTS] = self._costs()
        conf_int[_PRIORITY] = conf_int[_LOWER] - conf_int[_COSTS]

        return conf_int

    def _costs(self) -> pd.DataFrame:
        """Удельные торговые издержки.

        Полностью распределяются на покупаемую позицию с учетом ее последующего закрытия. Состоят из
        двух составляющих - комиссии и воздействия на рынок. Для учета воздействия на рынок
        используется Rule of thumb, trading one day’s volume moves the price by about one day’s
        volatility

        https://arxiv.org/pdf/1705.00109.pdf

        Размер операций на покупку условно выбран равным текущему кэшу, а на последующую продажу
        текущая позиция плюс кэш за вычетом уже учтенных издержек на продажу текущей позиции.

        Было решено отказаться от расчета производной так как для нулевых позиций издержки воздействия
        небольшие, но быстро нарастают с объемом. Расчет для условной сделки в размере кэша сразу
        отсекает совсем неликвидных кандидатов на покупку.
        """
#        return 0  #sn_added 202301


        port = self._portfolio

        cash = port.weight[CASH] / port.turnover_factor
        weight = port.weight / port.turnover_factor
        weight_cash = weight + cash

        impact_scale = 1.5

        return (
            (
                # Обычные издержки в две стороны
                config.COSTS * 2
                # Дневное СКО
                + (self.metrics.std / config.YEAR_IN_TRADING_DAYS**0.5)
                # Зависимость общих издержек от воздействия пропорционален степени 1.5 от нормированного на
                # дневной оборот объема. Совершается покупка на кэш сейчас и увеличиваются издержки на
                # ликвидацию позиции
                * (cash**impact_scale + (weight_cash**impact_scale - weight**impact_scale))
                # Делим на объем операции для получения удельных издержек
                / cash
            )
            # Умножаем на коэффициент пересчета в годовые значения
            * (config.YEAR_IN_TRADING_DAYS / config.FORECAST_DAYS)
            # Уменьшаем издержки в годовом выражении, если торговля идет не каждый день
            / config.TRADING_INTERVAL
        )


def _grad_conf_int(forecasts, p_value) -> tuple[float, float]:
    forecasts = (forecasts,)
    interval = stats.bootstrap(
        forecasts,
        np.median,
        confidence_level=(1 - p_value),
        random_state=0,
    ).confidence_interval

    return interval.low, interval.high
