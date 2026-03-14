#!/usr/bin/env python3
import os
import argparse
import logging

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from src import create_graph
from src.state import AgentState

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _get_llm(model_name: str | None = None):
    if os.getenv("GEMINI_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-3-flash-preview",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0)


def main():
    parser = argparse.ArgumentParser(description="PR Agent: natural language task → code changes → PR")
    parser.add_argument("--task", "-t", type=str, required=True, help="Natural language description (include repo URL in task if needed)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model override (e.g. gemini-2.0-flash, gpt-4o-mini)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps (default 50)")
    args = parser.parse_args()

    llm = _get_llm(args.model)
    graph = create_graph(llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "1"}}

    initial: AgentState = {
        "messages": [HumanMessage(content=args.task)],
    }

    for step, chunk in enumerate(graph.stream(initial, config)):
        if step >= args.max_steps:
            print("\n[Max steps reached]")
            break
        for node_name, value in chunk.items():
            if node_name in ("coder", "git_ops") and "messages" in value:
                last = value["messages"][-1]
                if hasattr(last, "content") and last.content:
                    print(f"[{node_name}]", last.content[:500] + ("..." if len(last.content) > 500 else ""))
            if node_name == "tester" and "tests_ok" in value:
                print(f"[tester] tests_ok={value.get('tests_ok')}")
            if node_name == "action" and "messages" in value:
                for m in value["messages"]:
                    if hasattr(m, "name"):
                        print(f"  [tool] {m.name}")

    state = graph.get_state(config)
    if state.values.get("messages"):
        final = state.values["messages"][-1]
        if hasattr(final, "content") and final.content:
            print("\n--- Final response ---\n")
            print(final.content)


if __name__ == "__main__":
    main()
