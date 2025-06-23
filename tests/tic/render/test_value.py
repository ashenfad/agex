from tic.eval.functions import UserFunction
from tic.eval.objects import TicDataClass, TicObject
from tic.render.value import ValueRenderer


def test_render_primitives():
    renderer = ValueRenderer()
    assert renderer.render(123) == "123"
    assert renderer.render("hello") == "'hello'"
    assert renderer.render(True) == "True"
    assert renderer.render(None) == "None"


def test_render_string_truncation():
    renderer = ValueRenderer(max_len=10)
    assert renderer.render("a" * 5) == "'aaaaa'"
    assert renderer.render("a" * 20) == "'aaaaaaaaaa...'"


def test_render_list_depth_limit():
    renderer = ValueRenderer(max_depth=1)
    nested_list = [1, [2, [3]]]
    assert renderer.render(nested_list) == "[1, [... (2 items)]]"


def test_render_list_length_limit():
    renderer = ValueRenderer(max_len=20)
    long_list = list(range(10))
    # Expect it to cut off around 4 or 5 elements
    assert renderer.render(long_list) == "[0, 1, 2, 3, ... (6 more)]"


def test_render_dict_depth_limit():
    renderer = ValueRenderer(max_depth=1)
    nested_dict = {"a": 1, "b": {"c": 2}}
    assert renderer.render(nested_dict) == "{'a': 1, 'b': {... (1 items)}}"


def test_render_tic_objects():
    renderer = ValueRenderer()
    # Mock a UserFunction
    mock_fn = UserFunction(name="my_func", args=None, body=[], closure_state=None)
    assert renderer.render(mock_fn) == "<function my_func>"

    # Mock a TicObject
    mock_cls = TicDataClass(name="MyData", fields=["x", "y"])
    mock_obj = TicObject(cls=mock_cls, attributes={"x": 1, "y": "hello"})
    assert renderer.render(mock_obj) == "MyData(x=1, y='hello')"


def test_render_tic_object_truncation():
    renderer = ValueRenderer(max_depth=1, max_len=40)
    inner_cls = TicDataClass(name="Inner", fields=["val"])
    inner_obj = TicObject(cls=inner_cls, attributes={"val": [1, 2, 3]})

    outer_cls = TicDataClass(name="Outer", fields=["a", "b", "c"])
    outer_obj = TicObject(
        cls=outer_cls,
        attributes={"a": inner_obj, "b": "a_long_string_value", "c": 3},
    )

    # Depth truncation on the nested object, length truncation on the second attribute
    assert renderer.render(outer_obj) == "Outer(a=Inner(val=[... (3 items)]), ...)"


def test_render_opaque_objects():
    renderer = ValueRenderer()

    class MyObject:
        pass

    class SizedObject:
        def __len__(self):
            return 10

    class ShapedObject:
        shape = (100, 200)

    assert renderer.render(MyObject()) == "<MyObject object>"
    assert renderer.render(SizedObject()) == "<SizedObject len=10>"
    assert renderer.render(ShapedObject()) == "<ShapedObject shape=(100, 200)>"
