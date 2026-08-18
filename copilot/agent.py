"""
copilot/agent.py — LangGraph ReAct agent for the MDT AI Copilot.

Uses create_react_agent from LangGraph v1.0 (stable, October 2025).
NOT AgentExecutor (deprecated), NOT a hand-rolled loop.

LLM Provider: OpenRouter (openai/gpt-oss-20b:free) via ChatOpenAI-compatible client.
Embeddings:   Cohere embed-v4.0 (used in rag/build_index.py).

The agent has two guardrails (Section 7.4 of PROJECT_GUIDE.md):
  1. Max tool calls per question: MAX_TOOL_CALLS (default 8, from .env)
  2. Tool output truncation: handled per-tool in copilot/tools.py

Usage:
    from copilot.agent import get_agent
    agent = get_agent()
    result = agent.invoke({"messages": [("human", "How many anomalies in cell 1?")]})
    print(result["messages"][-1].content)
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))

# ---------------------------------------------------------------------------
# System prompt (Section 7.4 of PROJECT_GUIDE.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a 5G RAN coverage analyst assistant for an MDT (Minimization of Drive Tests) AI Copilot system.

## Your tools:
- **query_reports** / **run_anomaly_scan** / **get_cell_summary** / **suggest_threshold**: Use for DATA questions about MDT samples, signal strength (RSRP), anomalies, and cell health.
- **retrieve_docs**: Use for CONCEPTUAL questions about how MDT works, 3GPP standards, C data structures, system design decisions, ring buffer behaviour, RSRP conversion formulas, trigger reasons, and why design choices were made.

## Critical rules:
1. Always cite specific numbers from tool results — never estimate or guess data values.
2. Always state whether the underlying data is synthetic (from the generator) or from a real capture. The synthetic data has an `is_injected_anomaly` ground-truth field.
3. For questions that need both data and explanation, use the appropriate tool for each part.
4. If a tool returns an error or "not found", say so explicitly rather than guessing.
5. Keep answers concise and grounded in tool results.

## Context:
- The ring buffer size is NR_MDT_LOG_BUFFER_SIZE = 64 entries per UE connection.
- Default thresholds in the gNB: rsrp_drop trigger = 3 dB, low_rsrp trigger = -100 dBm.
- RSRP index to dBm: RSRP_dBm = RSRP_index + 157 (3GPP standard).
- Synthetic data uses 10 UEs, 1 hour at 5-second intervals (seed=42).
"""

# ===========================================================================
# LLM Provider Configuration
# ===========================================================================

def _get_llm():
    """Return the LLM instance to power the agent."""
    import os
    from langchain_cohere import ChatCohere

    # Use Cohere's Command-R model which is free for developers
    # and natively supports excellent tool calling!
    return ChatCohere(
        model="command-r-plus-08-2024",
        cohere_api_key=os.environ.get("COHERE_API_KEY"),
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_agent():
    """Build and return the LangGraph ReAct agent (cached after first call).

    Uses create_react_agent with:
    - openai/gpt-oss-20b:free via OpenRouter (ChatOpenAI-compatible client)
    - All 5 tools from copilot/tools.py
    - recursion_limit set to MAX_TOOL_CALLS × 3 (each tool call ≈ 3 graph steps)
    """
    from langgraph.prebuilt import create_react_agent
    from copilot.tools import ALL_TOOLS

    llm = _get_llm()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def ask(question: str, verbose: bool = False) -> str:
    """Ask the agent a question and return the final answer as a string.

    Args:
        question: The natural-language question from the user.
        verbose:  If True, print tool calls as they happen.

    Returns:
        The agent's final answer string.
    """
    agent = get_agent()

    config = {
        "recursion_limit": MAX_TOOL_CALLS * 3,  # each call = ~3 graph steps
    }

    result = agent.invoke(
        {"messages": [("human", question)]},
        config=config,
    )

    messages = result.get("messages", [])

    if verbose:
        for msg in messages[1:]:  # skip the human message
            msg_type = type(msg).__name__
            if "ToolMessage" in msg_type:
                print(f"\n  [TOOL] {msg.name}: {str(msg.content)[:200]}...")
            elif "AIMessage" in msg_type and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n  [CALL] {tc['name']}({tc.get('args', {})})")

    # Final answer is the last AIMessage content
    for msg in reversed(messages):
        msg_type = type(msg).__name__
        if "AIMessage" in msg_type:
            content = msg.content
            # OpenAI-style: content is always a plain string
            if isinstance(content, str):
                return content
            # Fallback: list of text blocks (Anthropic-style — kept for safety)
            if isinstance(content, list):
                text_parts = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                    if not (isinstance(c, dict) and c.get("type") == "tool_use")
                ]
                return "\n".join(p for p in text_parts if p.strip())
            return str(content)

    return "No response generated."
