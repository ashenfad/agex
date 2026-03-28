"""Tests for collapse_same_role_messages."""

from agex.render.primitives import collapse_same_role_messages


class TestCollapseSameRoleMessages:
    def test_empty_list(self):
        assert collapse_same_role_messages([]) == []

    def test_single_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_alternating_roles_unchanged(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 3

    def test_str_str_merge(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "first\nsecond"

    def test_str_list_merge(self):
        msgs = [
            {"role": "user", "content": "text"},
            {"role": "user", "content": [{"type": "image", "image_data": "abc"}]},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "text"},
            {"type": "image", "image_data": "abc"},
        ]

    def test_list_str_merge(self):
        msgs = [
            {"role": "user", "content": [{"type": "image", "image_data": "abc"}]},
            {"role": "user", "content": "text"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "image", "image_data": "abc"},
            {"type": "text", "text": "text"},
        ]

    def test_list_list_merge(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "image", "image_data": "b"}]},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "image", "image_data": "b"},
        ]

    def test_three_consecutive(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "a\nb\nc"

    def test_mixed_runs(self):
        msgs = [
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "u1\nu2"}
        assert result[1] == {"role": "assistant", "content": "a1\na2"}
        assert result[2] == {"role": "user", "content": "u3"}

    def test_assistant_collapse(self):
        msgs = [
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        result = collapse_same_role_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "first\nsecond"

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        original = [dict(m) for m in msgs]
        collapse_same_role_messages(msgs)
        assert msgs == original
