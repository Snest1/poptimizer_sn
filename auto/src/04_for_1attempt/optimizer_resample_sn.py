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




    def __str__(self) -> str:
        """Информация о позициях, градиенты которых значимо отличны от 0."""

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



### buy and sell
        if 1 == 1:

            zero_buying = ['LQDT',
'AKMB', 'OBLG', 'SBGB', 'SBRB',                      # временно, так как без них заваливает UPRO
'GLDRUB_TOM', 'GOLD', 'BTCRUB', 'ETHRUB',
]

            skip_buying = [  #['LQDT', 'OBLG']
'AKME', #	БПИФ	БПИФ «Альфа-Капитал Управляемые российские акции»	0	Исключать		
'DIVD', #	БПИФ	БПИФ «ДОХОДЪ Индекс дивидендных акций РФ»	2428,2	Исключать		
'EQMX', #	БПИФ	ВТБ – Индекс МосБиржи	34587	Исключать		
'RCMX', #	БПИФ	БПИФ «Райффайзен – Индекс МосБиржи полной доходности 15»	2849,4	Исключать		
'SBMX', #	БПИФ	БПИФ «Индекс МосБиржи полной доходности «брутто»» 	17673,8	Исключать		
'SBRI', #	БПИФ	БПИФ «Сбер – Ответственные инвестиции»	0	Исключать		
'TMOS', #	БПИФ	БПИФ «Тинькофф Индекс Мосбиржи»	60830,42	Исключать		
'TRUR', #	БПИФ	БПИФ «Тинькофф – Стратегия вечного портфеля в рублях»	676479,44	Исключать		
'KOGK', #	Говно	Коршуновский ГОК	36800	Исключать	Низкая ликвидность	Дочка Мечела без дивов
'KUZB', #	Говно		882,5	Исключать	Низкая ликвидность	
'LNZL', #	Говно	ЛенЗолото	0	Исключать	Делистинг	
'LNZLP', #	Говно	ЛенЗолото	0	Исключать	Делистинг	
'NSVZ', #	Говно	Наука и связь	1400	Исключать	Низкая ликвидность	
'RBCM', #	Говно		856,2	Исключать	Параша на графике, низкая ликвидность	
'RUGR', #	Говно	РусГрэйн	4880	Исключать	Риск банкротства, делистинга	
'SFIN', #	Говно	SFI Инв.холдинг	13968	Исключать	Параша на графике	Инв.холдинг
'TGKD', #	Говно	Квадра	1285	Исключать	Низкая ликвидность, хаи	
'VLHZ', #	Говно	Владимирский ХЗ	731,5	Исключать	Параша на графике, низкая ликвидность	
'APTK',
'VRSB',			#Низкая ликвидность
'STSBP',		#Низкая ликвидность

# 'AKMB', 'OBLG', 'SBGB', 'SBRB',
]

            skip_selling = [] #['LQDT', 'OBLG']
            zero_selling = ['SLVRUB_TOM', 'GLDRUB_TOM', 'PLZL', 'YNDX', 'LQDT',
'BTCRUB', 'FIVE',
'SBGB',     # У меня уже лишку, но продавать только на ИИС  без вывода. Пометил красным - буду отслеживать, когда предложит продавать.
] #


            start_cash = 0
#            start_cash = 150000
            deal_size = 900
#            deal_size = 300
            allow_sell = False
            daily_limit = 100000
            allow_daily_relimit = True


            cur_cash = start_cash
            cur_daily_limit = 0
            daily_limits = 0
            hop_buy = 0
            hop_sell = 0


            self._logger.info(f'OPER\thop_b\thop_s\tcur_daily_limit\tdaily_limits\tbuy\tb_PRIORITY\tlots\tlots_buy\tsumm_b\tsell\ts_PRIORITY\tlots\tlots_sell\tsumm_s\tcash_before')

# main loop.
            while 1 == 1:

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
                while buy_index == '':
                    if rec.SIGNAL[skipping] == 'BUY' and rec.index[skipping] not in skip_buying:
                        buy_index = rec.index[skipping]
                    elif rec.index[skipping] in skip_buying:
                        skipping = skipping + 1
                    else:
                        self._logger.info(f'Nothing to buy. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f}')
                        quit()	## nothing to buy


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
                    if buy_index not in zero_buying:
                        cur_cash = cur_cash - summ_b
                    else:
                        self._logger.info(f'Zero buy {buy_index} {lots_b:.0f} lot(s) on {summ_b:.2f} (virtual). Now {rec.lots.loc[buy_index]} lot(s).')
                    hop_sell = 0
                    while cur_daily_limit >= daily_limit:
                        daily_limits = daily_limits + 1
                        self._logger.info(f'Daily limit reached. Cash = {cur_cash:.2f}  Daily_limit = {daily_limit:.2f} * {daily_limits}')
                        if daily_limits > 0 and allow_daily_relimit != True:
                            quit()
                        cur_daily_limit = cur_daily_limit - daily_limit


                elif allow_sell != True:     #no cash
                    self._logger.info(f'Cash is over. Cash = {cur_cash:.2f}  Cur_daily_limit = {cur_daily_limit:.2f} * {daily_limits}')
                    quit()
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
                            quit()

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
                    if sell_index not in zero_selling:
                        cur_cash = cur_cash + summ_s
                    else:
                        self._logger.info(f'Zero sell {sell_index} {lots_s:.0f} lot(s) on {summ_s:.2f} (virtual). Now {rec.lots.loc[sell_index]} lot(s).')


                cur_prot = self._update_portfolio(rec, 1)
                self._portfolio = cur_prot
                self._metrics = metrics.MetricsResample(self._portfolio)


            self._logger.info(f'Dummy quit')
            quit()


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
