from .helpers import eval_and_get_state


def test_evaluator_stress_test():
    """
    A comprehensive stress test that uses a wide variety of features together.
    """
    program = """
from dataclasses import dataclass

# 1. Define a dataclass for raw data
@dataclass
class Config:
    name: str
    value: int

# 2. Define a regular class to encapsulate logic
class Widget:
    def __init__(self, config):
        self.name = config.name
        self._value = config.value
        self.log = []
        self.log_message(f"Initialized with {self._value}")

    def increment(self, amount):
        self._value = self._value + amount
        self.log_message(f"Incremented to {self._value}")

    def get_value(self):
        return self._value

    def log_message(self, msg):
        self.log = self.log + [msg]

# 3. A factory function that uses both classes
def create_widget(name, val):
    conf = Config(name=name, value=val)
    return Widget(conf)

# 4. Main script logic
widgets = []
total = 0
names = ["a", "b", "c"]
for i in range(len(names)):
    w = create_widget(names[i], i * 10)
    w.increment(i + 1)
    widgets = widgets + [w]
    total = total + w.get_value()

# 5. Introspection
help(Widget)
dir_result = dir(widgets[0])
log_result = widgets[0].log

# 6. Error handling
try:
    widgets[0].non_existent_method()
except Exception as e:
    error_msg = "Caught expected error"

"""
    state = eval_and_get_state(program)

    # Check final state of variables
    assert state.get("total") == 36
    assert state.get("error_msg") == "Caught expected error"

    # Check introspection results in stdout
    stdout = state.get("__stdout__")
    assert len(stdout) == 2  # help() and dir() both print
    help_text = stdout[0][0]
    assert "Help on class Widget" in help_text
    assert "get_value(self)" in help_text

    # Check introspection results in variables
    dir_result = state.get("dir_result")
    log_result = state.get("log_result")
    assert isinstance(dir_result, list)
    assert sorted(dir_result) == [
        "__init__",
        "_value",
        "get_value",
        "increment",
        "log",
        "log_message",
        "name",
    ]
    assert log_result == ["Initialized with 0", "Incremented to 1"]

    # Check the state of the created widgets
    widgets = state.get("widgets")
    assert len(widgets) == 3
    assert widgets[2].attributes["name"] == "c"
