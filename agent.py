"""Phase 2 — Career agent: LangGraph ReAct + Gemini over graph tools (GraphRAG loop)."""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a Career Navigator assistant. You NEVER guess skills, roles or resources \
from training data — every factual claim must come from a tool result (the Neo4j knowledge graph).

Workflow per user message:
1. Identify the learner. If you don't know who they are, ask for their name. \
If they are new (tool says unknown user), register them with register_learner: ask for \
current skills + career goal first, then register, then analyze.
2. Retrieve with tools BEFORE answering:
   - gap / "what am I missing" / learning path -> get_learner_profile, then \
analyze_skill_gap + get_learning_path (cite prerequisites and core-vs-bonus).
   - "which role fits me" / compare -> compare_roles.
   - "I finished / learned X" -> save_skill, then re-run analyze_skill_gap to show new state.
   - "tell me about X skill" / resources -> explore_skill.
   - "what roles exist" -> list_roles.
3. Answer concisely in chat style: short gap summary, ordered phases, one resource per step. \
Always name the graph relationships you used (requires, prerequisite, teaches).

If the graph has no data for the question, say so instead of inventing.
Default demo user, if the UI sends none, is Aarav.
"""

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
        _agent = create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT,
                                    checkpointer=MemorySaver())
    return _agent


def chat(user_text: str, thread_id: str = "demo") -> str:
    """One chat turn. thread_id keeps per-user conversation memory."""
    print("[chat] user={} msg={}".format(thread_id, user_text[:120]), flush=True)
    agent = get_agent()
    final = None
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            msgs = update.get("messages", []) if isinstance(update, dict) else []
            for m in msgs:
                mtype = getattr(m, "type", "")
                if mtype == "ai" and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        print("[tool] {}({})".format(tc.get("name"), tc.get("args")), flush=True)
                elif mtype == "tool":
                    print("[tool-result] {}: {} chars".format(
                        getattr(m, "name", "?"), len(str(getattr(m, "content", "")))), flush=True)
                elif mtype == "ai" and getattr(m, "content", ""):
                    final = m.content
    print("[chat] done user={}".format(thread_id), flush=True)
    return final or "Sorry, I could not generate a response."
