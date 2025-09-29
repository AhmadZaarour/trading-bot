from engine.broker import BinanceFuturesBroker
from engine.data import BinanceDataProvider
from strategy.basic_strategy import MyStrategy
from engine.engine import Engine
from engine.risk import RiskManager
from backtest.runner import Backtester

def main():
    broker = BinanceFuturesBroker(testnet=True)
    data = BinanceDataProvider(testnet=True)
    risk = RiskManager(broker)
    
    strategy = MyStrategy()

    engine = Engine(data, broker, risk, strategy)
    backtest = Backtester(risk, broker, strategy, data)

    choice = input("1 for live trade, 2 for simulation:")
    while choice not in ['1', '2']:
        choice = input("Invalid choice. Please enter 1 for live trade, 2 for simulation:")
    
    if choice == '2':
        backtest.run()
    else:
        engine.run()

if __name__ == "__main__":
    main()

