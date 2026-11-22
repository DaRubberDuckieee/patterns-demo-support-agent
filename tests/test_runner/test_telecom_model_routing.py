import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import unit_provider

OPUS = "claude-opus-4-5"
GPT = "openai/gpt-5.2-2025-12-11"


@pytest.mark.parametrize(
    ("domain", "enabled", "agent", "model", "expected"),
    [
        ("telecom", True, "llm_agent", OPUS, GPT),
        ("telecom", True, "llm_agent", "anthropic/claude-opus-4-5-20251101", GPT),
        ("telecom", False, "llm_agent", OPUS, OPUS),
        ("retail", True, "llm_agent", OPUS, OPUS),
        ("telecom", True, "llm_agent_gt", OPUS, OPUS),
        ("telecom", True, "llm_agent", "gpt-4.1", "gpt-4.1"),
    ],
)
def test_telecom_model_routing(domain, enabled, agent, model, expected):
    config = TextRunConfig(
        domain=domain,
        agent=agent,
        llm_agent=model,
        route_telecom_troubleshooting=enabled,
    )

    assert config.effective_agent_model == expected
    assert unit_provider(config) == ("openai" if expected == GPT else None)
