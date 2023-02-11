import poptimizer.tvDatafeed
tv = poptimizer.tvDatafeed.TvDatafeed()

print(
    tv.get_hist(
        "BTCRUB",
        "BINANCE",
        interval=poptimizer.tvDatafeed.Interval.in_daily,
        n_bars=91,
        extended_session=False,
        adj_divi=False,
    )
)
