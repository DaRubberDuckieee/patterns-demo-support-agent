"""Offline coverage for telecom model selection and conversation fallback."""

from unittest.mock import Mock

import pytest

from tau2 import config
from tau2.agent import llm_agent
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, create_llm_agent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.domains.telecom.environment import get_environment
from tau2.runner.build import build_agent

CONFIGURED_MODEL = "claude-opus-4-5"
TELECOM_MODEL = "gpt-5.2-2025-12-11"


@pytest.fixture(autouse=True)
def routing_config(monkeypatch):
    monkeypatch.setattr(config, "TELECOM_LLM_ROUTING_ENABLED", True)
    monkeypatch.setattr(config, "TELECOM_LLM_AGENT", TELECOM_MODEL)


@pytest.fixture
def generate(monkeypatch):
    mock = Mock(return_value=AssistantMessage(role="assistant", content="Hello"))
    monkeypatch.setattr(llm_agent, "generate", mock)
    return mock


def make_agent(domain="telecom"):
    return create_llm_agent(
        tools=[],
        domain_policy="Identify the customer before troubleshooting.",
        llm=CONFIGURED_MODEL,
        llm_args={"temperature": 0},
        domain=domain,
    )


def next_turn(agent, state):
    return agent.generate_next_message(
        UserMessage(role="user", content="My mobile data is slow."), state
    )


def models(generate):
    return [call.kwargs["model"] for call in generate.call_args_list]


def test_runner_passes_telecom_domain_to_agent(generate):
    agent = build_agent("llm_agent", get_environment(), llm=CONFIGURED_MODEL, llm_args={})
    next_turn(agent, agent.get_init_state())
    assert models(generate) == [TELECOM_MODEL]
    assert agent.llm == CONFIGURED_MODEL


@pytest.mark.parametrize("domain", ["airline", "retail", "mock", None])
def test_other_domains_keep_configured_model(domain, generate):
    agent = make_agent(domain)
    next_turn(agent, agent.get_init_state())
    assert models(generate) == [CONFIGURED_MODEL]


def test_direct_constructor_without_domain_keeps_configured_model(generate):
    agent = LLMAgent(tools=[], domain_policy="Policy", llm=CONFIGURED_MODEL)
    next_turn(agent, agent.get_init_state())
    assert models(generate) == [CONFIGURED_MODEL]


@pytest.mark.parametrize("failure", ["exception", "none", "empty", "whitespace"])
def test_fallback_sticks_to_configured_model_for_conversation(failure, generate):
    agent = make_agent()
    state = agent.get_init_state()
    failed = {
        "exception": RuntimeError("provider unavailable"),
        "none": None,
        "empty": AssistantMessage(role="assistant", content="", cost=0.01),
        "whitespace": AssistantMessage(role="assistant", content=" \n\t"),
    }[failure]
    reply = AssistantMessage(role="assistant", content="How can I help?", cost=0.02)
    generate.side_effect = [failed, reply, reply]

    result, state = next_turn(agent, state)
    assert result is reply
    assert state.telecom_fallback_active
    assert len(state.messages) == 2  # one input and one accepted reply
    if failure == "empty":
        assert result.cost == pytest.approx(0.03)
    # Checkpoint restoration preserves the selected fallback.
    restored = LLMAgentState.model_validate_json(state.model_dump_json())
    next_turn(agent, restored)
    assert models(generate) == [TELECOM_MODEL, CONFIGURED_MODEL, CONFIGURED_MODEL]
    first, fallback = generate.call_args_list[:2]
    assert first.kwargs["messages"] == fallback.kwargs["messages"]
    assert first.kwargs["tools"] == fallback.kwargs["tools"]
    assert first.kwargs["temperature"] == fallback.kwargs["temperature"] == 0
    # A new conversation on the same agent tries the cheaper model again.
    generate.side_effect = None
    next_turn(agent, agent.get_init_state())
    assert models(generate)[-1] == TELECOM_MODEL


def test_tool_only_response_does_not_trigger_fallback(generate):
    reply = AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id="lookup",
                name="get_customer_by_phone",
                arguments={"phone_number": "555-123-2002"},
            )
        ],
    )
    generate.return_value = reply
    agent = make_agent()
    result, state = next_turn(agent, agent.get_init_state())
    assert result is reply
    assert not state.telecom_fallback_active
    assert models(generate) == [TELECOM_MODEL]


def test_fallback_failure_propagates_without_another_retry(generate):
    generate.side_effect = [
        RuntimeError("cheap failed"),
        RuntimeError("fallback failed"),
    ]
    agent = make_agent()
    state = agent.get_init_state()
    with pytest.raises(RuntimeError, match="fallback failed"):
        next_turn(agent, state)
    assert state.telecom_fallback_active
    assert len(state.messages) == 1
    assert models(generate) == [TELECOM_MODEL, CONFIGURED_MODEL]


def test_empty_fallback_fails_without_retrying(generate):
    generate.return_value = AssistantMessage(role="assistant", content="")
    agent = make_agent()
    with pytest.raises(ValueError, match="fallback model returned an empty response"):
        next_turn(agent, agent.get_init_state())
    assert models(generate) == [TELECOM_MODEL, CONFIGURED_MODEL]


def test_routing_off_keeps_existing_behavior(monkeypatch, generate):
    monkeypatch.setattr(config, "TELECOM_LLM_ROUTING_ENABLED", False)
    agent = make_agent()
    # Even today's empty-response behavior is preserved when routing is off.
    generate.return_value = AssistantMessage(role="assistant", content="")
    _, state = next_turn(agent, agent.get_init_state())
    assert not state.telecom_fallback_active
    assert models(generate) == [CONFIGURED_MODEL]
    generate.side_effect = RuntimeError("configured failed")
    with pytest.raises(RuntimeError, match="configured failed"):
        next_turn(agent, state)
    assert models(generate) == [CONFIGURED_MODEL, CONFIGURED_MODEL]


def test_telecom_model_is_configurable(monkeypatch, generate):
    monkeypatch.setattr(config, "TELECOM_LLM_AGENT", "custom-telecom-model")
    agent = make_agent()
    next_turn(agent, agent.get_init_state())
    assert models(generate) == ["custom-telecom-model"]


def test_same_configured_model_is_not_retried(generate):
    agent = create_llm_agent(
        tools=[], domain_policy="Policy", llm=TELECOM_MODEL, domain="telecom"
    )
    generate.side_effect = RuntimeError("provider failed")
    with pytest.raises(RuntimeError, match="provider failed"):
        next_turn(agent, agent.get_init_state())
    assert models(generate) == [TELECOM_MODEL]
