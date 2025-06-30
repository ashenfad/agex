import random

import numpy as np

from agex import Agent

data_maker = Agent(name="data_maker", primer="You excel at generating data via numpy.")

data_maker.module(np, visibility="low")
data_maker.module(np.random, visibility="low")
data_maker.module(random, visibility="low")


@data_maker.task
def make_data(prompt: str) -> list[np.ndarray]:  # type: ignore[return-value]
    """Produce numpy arrays given the prompt."""
    pass


gen_data_request = """
Please generate 1000 numpy arrays, each with 120 elements. They should represent
two minute long processes when a manufacturing step happens and power is is monitored.
The signals should be dynamic yet similar with small variations.
Please make 50 of the signals have abberations.
"""


# data = make_data(gen_data_request)
