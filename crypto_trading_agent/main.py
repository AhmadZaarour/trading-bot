from engine.broker import BinanceFuturesBroker
from engine.data import BinanceDataProvider
from strategy.basic_strategy import MyStrategy
from engine.engine import Engine
from engine.risk import RiskManager

def main():
    broker = BinanceFuturesBroker(testnet=True)
    data = BinanceDataProvider(testnet=True)
    risk = RiskManager(broker)
    
    strategy = MyStrategy()

    engine = Engine(data, broker, risk, strategy)
    engine.run()

if __name__ == "__main__":
    main()

