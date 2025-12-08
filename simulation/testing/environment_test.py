# only run with command: python3 -m simulation.testing.environment_test

import logging
from simulation.engine.environment import Environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnvironmentTest")

class MockInstanceManager:
    def __init__(self, env):
        self.env = env
        self.started = False
    def run(self):
        self.started = True
        logger.info("InstanceManager run invoked")
        yield self.env.env.timeout(0.5)

def mock_spawner_factory(env, flag):
    def _spawner():
        flag["spawned"] = True
        logger.info("Spawner executed")
        yield env.env.timeout(0.2)
    return _spawner

def test_environment_initialization():
    env = Environment(config={"seed": 123})
    logger.info("Environment initialized in test")
    assert env.config["seed"] == 123
    assert env.now == 0

def test_environment_with_spawner_and_manager():
    env = Environment()
    flags = {"spawned": False}
    spawner = mock_spawner_factory(env, flags)
    env.register_spawner(spawner)
    manager = MockInstanceManager(env)
    env.register_instance_manager(manager)
    env.start(until=1.0)
    assert flags["spawned"] is True
    assert manager.started is True
    assert env.now == 1.0

def test_scheduling():
    env = Environment()
    flags = {"called": False}
    def callback():
        flags["called"] = True
        logger.info("Scheduled callback executed")
    env.schedule_in(0.5, callback)
    env.start(until=1.0)
    assert flags["called"] is True
    assert env.now == 1.0

def test_integration_test():
    env = Environment()
    assert env.integration_test() is True
