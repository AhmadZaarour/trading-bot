from engine.broker import BinanceFuturesBroker
from engine.data import BinanceDataProvider
from strategy.basic_strategy import MyStrategy
from engine.engine import Engine
from engine.risk import RiskManager
from backtest.runner import Backtester
from engine.engine_thrader import EngineThreader
import yaml

def main():
    broker = BinanceFuturesBroker(testnet=True)
    data = BinanceDataProvider(testnet=True)
    risk = RiskManager(broker)
    
    strategy = MyStrategy()
    
    # Open and load the YAML file
    with open("config/default.yaml", "r") as file:
        config = yaml.safe_load(file)


    engine = Engine(data, broker, risk, strategy, 
                    symbol=config["SYMBOL"][0],
                    interval=config["INTERVAL"],
                    lookback=config["LOOKBACK"],
                    max_leverage=config["RISK"]["MAX_LEVERAGE"],
                    risk_per_trade=config["RISK"]["RISK_PER_TRADE"],
                    max_bars_per_trade=config["RISK"]["MAX_BARS_PER_TRADE"],
                    pol_seconds=config["POLL_SECONDS"]
                   )
    engine_threader = EngineThreader(config, data, broker, risk, strategy)
    backtest = Backtester(risk, broker, strategy, data, config)

    choice = input("1 for live trade, 2 for simulation: ")
    while choice not in ['1', '2']:
        choice = input("Invalid choice. Please enter 1 for live trade, 2 for simulation: ")
    
    if choice == '2':
        backtest.run()
    else:
        engine.run()

if __name__ == "__main__":
    main()

