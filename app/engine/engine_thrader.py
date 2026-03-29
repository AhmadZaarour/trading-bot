import threading
import time
from typing import Dict, List
from datetime import datetime
from engine.engine import Engine

class EngineThreader(threading.Thread):
    def __init__(self, config, data, broker, risk, strategy):
        super().__init__()
        self.symbols = config["SYMBOL"]
        self.data = data
        self.broker = broker
        self.risk = risk
        self.strategy = strategy
        self.engines: Dict[str, Engine] = {}
        self.stop_event = threading.Event()

    def run(self):
        print("Starting EngineThreader...")
        for symbol in self.symbols:
            strategy = self.strategy()
            engine = Engine(
                data=self.data,
                broker=self.broker,
                risk=self.risk,
                strategy=strategy
            )
            self.engines[symbol] = engine
            threading.Thread(target=engine.run).start()
            print(f"Engine started for symbol: {symbol}")

        while not self.stop_event.is_set():
            time.sleep(1)

    def stop(self):
        print("Stopping EngineThreader...")
        self.stop_event.set()
        for symbol, engine in self.engines.items():
            print(f"Stopping engine for symbol: {symbol}")
            engine.stop()