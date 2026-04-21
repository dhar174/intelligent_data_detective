"""
Unit tests for MyChatOpenai payload mutations.

Validates without making real API calls:
- max_tokens → max_completion_tokens rename
- o-series model: system role → developer role
- non-o-series model: system role stays system
- Both mutations together
"""
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        if not idd_core.HAS_LANGCHAIN:
            pytest.skip("LangChain not available")
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


@pytest.fixture(scope="module")
def chat_instance(core):
    """Create a MyChatOpenai instance with a fake key (no real calls made)."""
    return core.MyChatOpenai(model="gpt-4o", api_key="sk-test-fake-key-for-unit-tests")


def _make_payload(messages=None, **extra):
    """Build a minimal payload dict like _get_request_payload_mod would return."""
    payload = {"model": "gpt-4o", "messages": messages or [], **extra}
    return payload


class TestMaxTokensRename:
    def test_max_tokens_renamed_to_max_completion_tokens(self, chat_instance, core):
        """max_tokens in payload must be renamed to max_completion_tokens."""
        raw = _make_payload(max_tokens=512)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = chat_instance._get_request_payload([])
        assert "max_completion_tokens" in result
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 512

    def test_no_max_tokens_key_untouched(self, chat_instance, core):
        """If max_tokens is absent, payload passes through unchanged."""
        raw = _make_payload()
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = chat_instance._get_request_payload([])
        assert "max_tokens" not in result
        assert "max_completion_tokens" not in result


class TestOSeriesRoleRename:
    def test_o_series_system_role_renamed_to_developer(self, core):
        """For o-series models, 'system' role is renamed 'developer'."""
        instance = core.MyChatOpenai(model="o3-mini", api_key="sk-test-fake")
        messages = [{"role": "system", "content": "You are helpful."}]
        raw = _make_payload(messages=messages)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = instance._get_request_payload([])
        roles = [m["role"] for m in result.get("messages", [])]
        assert "developer" in roles, f"Expected 'developer' role, got {roles}"
        assert "system" not in roles, f"'system' role should be gone, got {roles}"

    def test_o1_series_system_role_renamed(self, core):
        """o1 prefix also triggers the rename."""
        instance = core.MyChatOpenai(model="o1-preview", api_key="sk-test-fake")
        messages = [{"role": "system", "content": "Instructions."}]
        raw = _make_payload(messages=messages)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = instance._get_request_payload([])
        roles = [m["role"] for m in result.get("messages", [])]
        assert "developer" in roles

    def test_non_o_series_system_role_unchanged(self, chat_instance, core):
        """For gpt-4o, 'system' role stays as 'system'."""
        messages = [{"role": "system", "content": "You are helpful."}]
        raw = _make_payload(messages=messages)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = chat_instance._get_request_payload([])
        roles = [m["role"] for m in result.get("messages", [])]
        assert "system" in roles, f"'system' role should be preserved for gpt-4o, got {roles}"
        assert "developer" not in roles

    def test_user_role_not_affected_for_o_series(self, core):
        """Only 'system' roles are renamed; 'user' and 'assistant' are untouched."""
        instance = core.MyChatOpenai(model="o3-mini", api_key="sk-test-fake")
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi!"},
        ]
        raw = _make_payload(messages=messages)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = instance._get_request_payload([])
        roles = [m["role"] for m in result.get("messages", [])]
        assert "user" in roles
        assert "assistant" in roles
        assert "developer" in roles
        assert "system" not in roles

    def test_multiple_system_messages_all_renamed(self, core):
        """Multiple system messages all get renamed."""
        instance = core.MyChatOpenai(model="o3-mini", api_key="sk-test-fake")
        messages = [
            {"role": "system", "content": "First."},
            {"role": "system", "content": "Second."},
        ]
        raw = _make_payload(messages=messages)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = instance._get_request_payload([])
        roles = [m["role"] for m in result.get("messages", [])]
        assert roles.count("developer") == 2
        assert "system" not in roles


class TestCombinedMutations:
    def test_max_tokens_and_system_rename_both_applied(self, core):
        """Both max_tokens rename and system→developer apply together for o-series."""
        instance = core.MyChatOpenai(model="o3-mini", api_key="sk-test-fake")
        messages = [{"role": "system", "content": "Be helpful."}]
        raw = _make_payload(messages=messages, max_tokens=1024)
        with patch.object(
            core.MyChatOpenai, "_get_request_payload_mod", return_value=raw
        ):
            result = instance._get_request_payload([])
        assert "max_completion_tokens" in result
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 1024
        roles = [m["role"] for m in result.get("messages", [])]
        assert "developer" in roles
        assert "system" not in roles
