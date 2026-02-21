from agex.render.value import ValueRenderer


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
