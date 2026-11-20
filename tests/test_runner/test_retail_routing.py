"""Offline retail routing coverage; the runner test tier runs these in CI."""

import json
from unittest.mock import Mock

import pytest

from tau2 import config
from tau2.agent import retail_routing
from tau2.agent.llm_agent import LLMAgent, create_llm_agent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

CONFIGURED = "claude-opus-4-5"
ROUTINE = "gpt-5.2-2025-12-11"


def classification(types, confident=True):
    return AssistantMessage(
        role="assistant",
        content=json.dumps({"confident": confident, "request_types": types}),
    )


def response():
    return AssistantMessage(role="assistant", content="How can I help?")


def user(content="I want to return my shoes."):
    return UserMessage(role="user", content=content)


@pytest.fixture
def agent():
    return create_llm_agent(
        tools=[],
        domain_policy="Retail policy",
        llm=CONFIGURED,
        llm_args={"temperature": 0.3},
        domain="retail",
    )


@pytest.fixture
def generate(monkeypatch):
    mock = Mock()
    monkeypatch.setattr(retail_routing, "generate", mock)
    return mock


@pytest.mark.parametrize(
    ("text", "types"),
    [
        ("I want to return my shoes.", ["return"]),
        ("Exchange my shirt for the same shirt in blue.", ["variant_exchange"]),
        ("Change my shipping address.", ["address_change"]),
        ("Return my shoes and update my address.", ["return", "address_change"]),
    ],
)
def test_routine_uses_cheaper_model(agent, generate, text, types):
    generate.side_effect = [classification(types), response()]
    _, state = agent.generate_next_message(user(text), agent.get_init_state())
    assert [c.kwargs["model"] for c in generate.call_args_list] == [ROUTINE, ROUTINE]
    assert generate.call_args.kwargs["reasoning_effort"] == "none"
    assert "temperature" not in generate.call_args.kwargs
    assert len(state.messages) == 2  # Classifier messages never enter agent history.
    assert agent.llm == CONFIGURED


@pytest.mark.parametrize(
    "decision",
    [
        classification(["other"]),
        classification(["unknown"]),
        classification(["return"], confident=False),
        classification(["return", "other"]),
        classification([]),
        AssistantMessage(role="assistant", content="invalid JSON"),
        AssistantMessage(role="assistant", content='{"request_types": ["return"]}'),
        classification(["unsupported"]),
        RuntimeError("classifier unavailable"),
    ],
)
def test_nonroutine_or_uncertain_keeps_configured_model(agent, generate, decision):
    generate.side_effect = [decision, response()]
    agent.generate_next_message(user(), agent.get_init_state())
    assert generate.call_args.kwargs["model"] == CONFIGURED
    assert generate.call_args.kwargs["temperature"] == 0.3


def test_configured_types_and_model_are_respected(agent, generate, monkeypatch):
    monkeypatch.setattr(config, "RETAIL_ROUTINE_REQUEST_TYPES", {"address_change"})
    monkeypatch.setattr(config, "RETAIL_ROUTINE_MODEL", "another-routine-model")
    generate.side_effect = [classification(["return"]), response()]
    agent.generate_next_message(user(), agent.get_init_state())
    assert generate.call_args_list[0].kwargs["model"] == "another-routine-model"
    assert generate.call_args.kwargs["model"] == CONFIGURED
    generate.reset_mock()
    generate.side_effect = [classification(["address_change"]), response()]
    agent.generate_next_message(user("Update my address"), agent.get_init_state())
    assert generate.call_args.kwargs["model"] == "another-routine-model"


def test_routing_off_skips_classifier(agent, generate, monkeypatch):
    monkeypatch.setattr(config, "RETAIL_MODEL_ROUTING_ENABLED", False)
    generate.return_value = response()
    agent.generate_next_message(user(), agent.get_init_state())
    generate.assert_called_once()
    assert generate.call_args.kwargs["model"] == CONFIGURED
    assert generate.call_args.kwargs["temperature"] == 0.3


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("model unavailable"), AssistantMessage(role="assistant", content="")],
)
def test_failure_retries_once_and_stays_on_configured(agent, generate, failure):
    generate.side_effect = [classification(["return"]), failure, response(), response()]
    _, state = agent.generate_next_message(user(), agent.get_init_state())
    first_try, retry = generate.call_args_list[1:3]
    assert first_try.kwargs["messages"] is retry.kwargs["messages"]
    assert len(state.messages) == 2
    agent.generate_next_message(user("Yes, please"), state)
    assert [c.kwargs["model"] for c in generate.call_args_list] == [
        ROUTINE,
        ROUTINE,
        CONFIGURED,
        CONFIGURED,
    ]
    assert agent.get_init_state().retail_routing.fallback_used is False


def test_fallback_failure_propagates(agent, generate):
    generate.side_effect = [
        classification(["return"]),
        RuntimeError("routine failed"),
        RuntimeError("configured failed"),
    ]
    with pytest.raises(RuntimeError, match="configured failed"):
        agent.generate_next_message(user(), agent.get_init_state())
    assert generate.call_count == 3


def test_tools_keep_route_and_new_intent_is_reclassified(agent, generate):
    tool_reply = AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id="one", name="get_order", arguments={})],
    )
    generate.side_effect = [
        classification(["return"]),
        tool_reply,
        response(),
        classification(["return", "other"]),
        response(),
    ]
    _, state = agent.generate_next_message(user(), agent.get_init_state())
    agent.generate_next_message(
        ToolMessage(role="tool", id="one", content="Order details"),
        state,
    )
    assert generate.call_count == 3
    agent.generate_next_message(user("Also dispute this payment"), state)
    assert generate.call_args.kwargs["model"] == CONFIGURED
    assert "dispute" in generate.call_args_list[3].kwargs["messages"][1].content


def test_nonretail_uses_existing_generation_path(generate, monkeypatch):
    original_generate = Mock(return_value=response())
    monkeypatch.setattr("tau2.agent.llm_agent.generate", original_generate)
    agent = LLMAgent(
        tools=[], domain_policy="Airline", llm=CONFIGURED, domain="airline"
    )
    agent.generate_next_message(user(), agent.get_init_state())
    original_generate.assert_called_once()
    generate.assert_not_called()


@pytest.mark.parametrize("multiple", [False, True])
def test_tool_failure_continues_on_configured_model(agent, generate, multiple):
    generate.side_effect = [classification(["return"]), response(), response()]
    _, state = agent.generate_next_message(user(), agent.get_init_state())
    tool_result = ToolMessage(role="tool", id="failed", content="Failed", error=True)
    message = tool_result
    if multiple:
        message = MultiToolMessage(
            role="tool",
            tool_messages=[
                tool_result,
                ToolMessage(role="tool", id="ok", content="Success"),
            ],
        )
    agent.generate_next_message(message, state)
    assert generate.call_args.kwargs["model"] == CONFIGURED
    assert tool_result in generate.call_args.kwargs["messages"]
    assert state.retail_routing.fallback_used


def test_runner_passes_retail_domain_to_factory(monkeypatch):
    from tau2.runner import build

    factory = Mock()
    monkeypatch.setattr(build.registry, "get_agent_factory", lambda name: factory)
    monkeypatch.setattr(build, "agent_domain_policy", lambda *a, **k: "Retail")
    environment = Mock()
    environment.get_domain_name.return_value = "retail"
    environment.get_tools.return_value = []
    build.build_agent("llm_agent", environment, llm=CONFIGURED)
    assert factory.call_args.kwargs["domain"] == "retail"
    assert factory.call_args.kwargs["llm"] == CONFIGURED
