"""Запуск основных операций с помощью CLI."""
import logging

import typer

from poptimizer import config
from poptimizer.data.views import div_status
from poptimizer.evolve import Evolution
from poptimizer.portfolio import load_from_yaml, optimizer_hmean, optimizer_resample, optimizer_resample_sn
#from poptimizer.portfolio import load_from_yamlwl

LOGGER = logging.getLogger()


def evolve() -> None:
    """Run evolution."""
    ev = Evolution()
    ev.evolve()


def dividends(ticker: str) -> None:
    """Get dividends status."""
    div_status.dividends_validation(ticker)


def optimize(date: str = typer.Argument(..., help="YYYY-MM-DD"), for_sell: int = 1) -> None:
    """Optimize portfolio."""
    port = load_from_yaml(date)
#    portwl = load_from_yamlwl(date)

#SNEDIT
#    LOGGER.info(portwl)

    if config.OPTIMIZER == "resample":
        opt = optimizer_resample.Optimizer(port, for_sell=for_sell)
    elif config.OPTIMIZER == "resample_sn":
        opt = optimizer_resample_sn.Optimizer(port, for_sell=for_sell)
    else:
        opt = optimizer_hmean.Optimizer(port)
#        opt = optimizer_hmean.Optimizer(port, portwl)

    LOGGER.info(opt.portfolio)
    LOGGER.info(opt.metrics)
    LOGGER.info(opt)

    div_status.new_dividends(tuple(port.index[:-2]))


if __name__ == "__main__":
    app = typer.Typer(help="Run poptimizer subcommands.", add_completion=False)

    app.command()(evolve)
    app.command()(dividends)
    app.command()(optimize)

    app(prog_name="poptimizer")
