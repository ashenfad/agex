"""
Data Generation

Agent generates complex NumPy arrays representing synthetic datasets.
Creates bulk data (1000 arrays) that flows to user or other agents
without special handling.

Note: This example was tested with `gpt-4.1-nano`, highlighting how `agex`'s
"micro-DSL" approach—providing a focused set of capabilities—can guide even
smaller models to success on complex tasks.
"""

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


def main():
    data = make_data(gen_data_request)
    print(len(data))
    print(type(data[0]))
    print(data[0][:3])
    # 1000
    # <class 'numpy.ndarray'>
    # [ 0.16647177 -0.09266472  0.09004741]


if __name__ == "__main__":
    # Run with: python examples/data.py OR python -m examples.data
    main()
