from engine.broker import BinanceFuturesBroker
from engine.data import BinanceDataProvider, SpotDataProvider
from engine.spot_broker import SpotBroker
from strategy.basic_strategy import MyStrategy
from engine.engine import Engine
from engine.spot_engine import SpotEngine
from engine.risk import RiskManager
from backtest.runner import Backtester
from backtest.spot_simulation import SpotBacktester
from engine.engine_thrader import EngineThreader
import yaml


def main():
    strategy = MyStrategy()

    with open("config/default.yaml", "r") as file:
        config = yaml.safe_load(file)

    symbol = config["SYMBOL"][0] if isinstance(config["SYMBOL"], list) else config["SYMBOL"]

    futures_broker = BinanceFuturesBroker(testnet=True)
    futures_data = BinanceDataProvider(testnet=True)
    futures_risk = RiskManager(futures_broker)

    spot_broker = SpotBroker(testnet=True)
    spot_data = SpotDataProvider(testnet=True)

    futures_engine = Engine(
        futures_data,
        futures_broker,
        futures_risk,
        strategy,
        symbol=symbol,
        interval=config["INTERVAL"],
        lookback=config["LOOKBACK"],
        max_leverage=config["RISK"]["MAX_LEVERAGE"],
        risk_per_trade=config["RISK"]["RISK_PER_TRADE"],
        max_bars_per_trade=config["RISK"]["MAX_BARS_PER_TRADE"],
        pol_seconds=config["POLL_SECONDS"],
    )
    engine_threader = EngineThreader(config, futures_data, futures_broker, futures_risk, strategy)
    backtest = Backtester(futures_risk, futures_broker, strategy, futures_data, config)

    spot_engine = SpotEngine(
        spot_data,
        spot_broker,
        strategy,
        symbol=symbol,
        interval=config["INTERVAL"],
        lookback=config["LOOKBACK"],
        risk_per_trade=config["RISK"]["RISK_PER_TRADE"],
        max_bars_per_trade=config["RISK"]["MAX_BARS_PER_TRADE"],
        poll_seconds=config["POLL_SECONDS"],
    )
    spot_backtest = SpotBacktester(spot_broker, strategy, spot_data, config)

    choice = input("1 for futures live, 2 for futures backtest, 3 for spot live, 4 for spot backtest: ")
    while choice not in ["1", "2", "3", "4"]:
        choice = input("Invalid choice. Please enter 1, 2, 3, or 4: ")

    if choice == "2":
        backtest.run()
    elif choice == "3":
        spot_engine.run()
    elif choice == "4":
        spot_backtest.run()
    else:
        futures_engine.run()


if __name__ == "__main__":
    main()
