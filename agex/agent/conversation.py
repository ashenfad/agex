"""
Conversation logging utilities for storing and reconstructing message history in state.

This module provides functions for managing conversation logs as hidden variables
in versioned state, enabling conversation history to be captured, versioned, and
reconstructed from any state snapshot.
"""

from typing import Sequence

from agex.llm.core import Message, TextMessage
from agex.state import State


def add_message(state: State, message: Message) -> None:
    """Add a message to the conversation log in state.

    Args:
        state: The versioned state to store the message in
        message: The Message object to store
    """
    # Get current message log
    msg_log = state.get("__msg_log__", [])

    # Generate next message key
    next_msg_num = len(msg_log) + 1
    msg_key = f"__msg{next_msg_num}__"

    # Store message object directly
    state.set(msg_key, message)

    # Update message log
    msg_log.append(msg_key)
    state.set("__msg_log__", msg_log)


def conversation_log(state: State, system_message: str) -> Sequence[Message]:
    """Reconstruct the full conversation from state.

    Args:
        state: The state containing the conversation log
        system_message: The system message to prepend to the conversation

    Returns:
        List of Message objects representing the full conversation
    """
    msg_log = state.get("__msg_log__", [])
    conversation_messages = [
        state.get(msg_key) for msg_key in msg_log if state.get(msg_key)
    ]

    return [TextMessage(role="system", content=system_message)] + conversation_messages


def initialize_conversation_log(state: State) -> None:
    """Initialize conversation log if not exists.

    Args:
        state: The state to initialize
    """
    if "__msg_log__" not in state:
        state.set("__msg_log__", [])


def get_conversation_length(state: State) -> int:
    """Get the number of messages in the conversation log.

    Args:
        state: The state containing the conversation log

    Returns:
        Number of messages in the conversation
    """
    return len(state.get("__msg_log__", []))


def clear_conversation_log(state: State) -> None:
    """Clear the conversation log from state.

    Args:
        state: The state to clear
    """
    # Remove all message entries
    msg_log = state.get("__msg_log__", [])
    for msg_key in msg_log:
        state.remove(msg_key)

    # Clear the message log itself
    state.remove("__msg_log__")
