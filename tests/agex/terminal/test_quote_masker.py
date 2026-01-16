from agex.terminal.quote_masker import mask_quotes, unmask_and_unquote, unmask_quotes


def test_basic_masking():
    text = 'echo "hello world"'
    masked, mapping = mask_quotes(text)

    # Expect: echo __Q_...__
    assert "echo " in masked
    assert '"hello world"' not in masked
    assert len(mapping) == 1

    token = list(mapping.keys())[0]
    assert token in masked
    assert mapping[token] == '"hello world"'


def test_single_quotes():
    text = "grep 'foo bar' file.txt"
    masked, mapping = mask_quotes(text)

    assert "grep " in masked
    assert "'foo bar'" not in masked
    assert len(mapping) == 1
    assert mapping[list(mapping.keys())[0]] == "'foo bar'"


def test_mixed_quotes():
    text = "echo \"foo\" 'bar'"
    masked, mapping = mask_quotes(text)

    assert len(mapping) == 2

    # Unmasking should restore original
    restored = unmask_quotes(masked, mapping)
    assert restored == text


def test_escaped_quotes_inside():
    text = 'echo "foo \\" bar"'
    masked, mapping = mask_quotes(text)

    assert len(mapping) == 1
    original = mapping[list(mapping.keys())[0]]
    assert original == '"foo \\" bar"'

    restored = unmask_quotes(masked, mapping)
    assert restored == text


def test_unmask_and_unquote():
    text = 'grep "pattern"'
    masked, mapping = mask_quotes(text)

    token = list(mapping.keys())[0]
    result = unmask_and_unquote(token, mapping)

    assert result == "pattern"


def test_unmask_and_unquote_escapes():
    text = 'echo "foo \\" bar"'
    masked, mapping = mask_quotes(text)

    token = list(mapping.keys())[0]
    print(f"DEBUG: original={mapping[token]}")
    result = unmask_and_unquote(token, mapping)

    # Should strip quotes AND unescape the inner quote
    assert result == 'foo " bar'
