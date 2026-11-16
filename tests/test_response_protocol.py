"""Reasoning delimiters must not let a discarded candidate answer enter the score."""

import pytest

from pdbthink.evaluation.response_protocol import MARIN_THINKING, answer_content, response_tool_events


def test_marin_reasoning_candidates_are_not_answers():
    message = {"content": "<|start_think|>Candidate: FINAL: A<|end_think|>FINAL: B"}
    assert answer_content(message, MARIN_THINKING) == "FINAL: B"
    assert answer_content({"content": "<|start_think|>Candidate: FINAL: A"}, MARIN_THINKING) == ""


@pytest.mark.parametrize("content", [
    "<|end_think|>FINAL: A",
    "<|end_think|><|start_think|>FINAL: A",
    "Prefix<|start_think|>reason<|end_think|>FINAL: A",
    "<|start_think|>one<|end_think|><|start_think|>two<|end_think|>FINAL: A",
])
def test_malformed_delimiters_cannot_select_a_convenient_answer(content):
    assert answer_content({"content": content}, MARIN_THINKING) == ""


def test_markerless_answers_and_other_models_keep_their_scoring_input():
    message = {"content": "FINAL: A"}
    assert answer_content(message, MARIN_THINKING) == "FINAL: A"
    assert answer_content({"content": "<|start_think|>FINAL: A"}) == "<|start_think|>FINAL: A"


def test_delimited_tools_in_reasoning_are_audited():
    response = {"choices": [{"message": {
        "content": '<|start_think|><tool_call>{"name":"python"}</tool_call><|end_think|>FINAL: A'
    }}]}
    assert response_tool_events(response, MARIN_THINKING)


def test_unknown_protocol_fails_closed():
    with pytest.raises(ValueError, match="Unknown response protocol"):
        answer_content({"content": "FINAL: A"}, "unknown")
