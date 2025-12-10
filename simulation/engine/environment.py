import simpy
import random
import numpy as np
from typing import Callable, Dict, Optional, Any


class Environment:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._seed = self.config.get("seed", None)

        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)

        self.env = simpy.Environment()

        self.instance_manager = None
        self.spawners = []
        self.resource_calendars = {}

        self._started = False
        self._stopped = False

    # REGISTRATION
    def register_instance_manager(self, manager) -> None:
        self.instance_manager = manager

    def register_spawner(self, spawner: Callable) -> None:
        self.spawners.append(spawner)

    def register_resource_calendar(self, name: str, calendar) -> None:
        self.resource_calendars[name] = calendar

    #SCHEDULING
    def schedule_in(self, delay: float, callback: Callable, *args, **kwargs):
        def _event_process():
            yield self.env.timeout(delay)
            return callback(*args, **kwargs)

        return self.env.process(_event_process())

    def schedule_at(self, time_point: float, callback: Callable, *args, **kwargs):
        delay = max(0, time_point - self.env.now)
        return self.schedule_in(delay, callback, *args, **kwargs)

    def start_process(self, process_fn: Callable, *args, **kwargs):
        return self.env.process(process_fn(*args, **kwargs))

    # TIME
    @property
    def now(self) -> float:
        return self.env.now

    def advance(self, until: float):
        self.env.run(until=until)

    # LIFECYCLE
    def start(self, until: Optional[float] = None):
        if self._started:
            raise RuntimeError("Simulation environment already started")

        self._started = True

        for spawner in self.spawners:
            self.env.process(spawner())

        if self.instance_manager is not None:
            self.env.process(self.instance_manager.run())

        self.env.run(until=until)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.env.exit()

    # TEST
    def integration_test(self) -> bool:
        test_flag = {"ran": False}

        def _test_callback():
            test_flag["ran"] = True

        self.schedule_in(1.0, _test_callback)
        self.start(until=2.0)
        return test_flag["ran"]
