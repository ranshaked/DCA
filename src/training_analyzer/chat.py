"""Helpers for grounding chat turns in retrieved knowledge-base context."""

from __future__ import annotations

from training_analyzer.prompts import render_prompt


def messages_with_retrieved_context(
    messages: list[dict[str, str]],
    retrieved_context: str,
) -> list[dict[str, str]]:
    """Return a copy with retrieval context attached to the latest user turn."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("The latest chat message must be from the user.")

    augmented = [message.copy() for message in messages]
    question = augmented[-1]["content"]
    augmented[-1]["content"] = render_prompt(
        "chat_grounding",
        user_question=question,
        retrieved_context=retrieved_context,
    )
    return augmented
