__all__ = ["SparkSchedSimEnv"]

from gymnasium.envs.registration import register
from .spark_sched_sim import SparkSchedSimEnv

# Register both with and without namespace for compatibility
register(id="SparkSchedSimEnv-v0", entry_point="spark_sched_sim:SparkSchedSimEnv")
register(id="spark_sched_sim/SparkSchedSimEnv-v0", entry_point="spark_sched_sim:SparkSchedSimEnv")
