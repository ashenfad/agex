"""Agent-defined tasks: the agent's own code asks a helper for judgment.

    AGEX_MODEL=anthropic:claude-sonnet-5-5 python examples/agent_tasks.py

``agent_tasks`` grants ``agex`` to the workspace, and the session's
agent defines tasks with it in its scripts: here, one that labels a
review, mapped over a file of them. Each call is answered by a helper
agent that sees only the call, on a scratch world of its own.
"""

import json
import os

from nontainer import Profile, PythonConfig, Store

from agex import Agent, agent_tasks

REVIEWS = [
    "Arrived a day early and works perfectly.",
    "Broke within a week. Support never answered.",
    "Great sound, but the app keeps logging me out.",
    "Exactly as described. Would buy again.",
]

PROMPT = """\
reviews.json holds a list of product reviews. Define an agex task that
labels one review positive, negative or mixed, map it over the reviews,
and write summary.json: {"positive": n, "negative": n, "mixed": n}.
"""


def main(agent: Agent) -> None:
    profile = Profile(python=PythonConfig(host_objects={"agex": agent_tasks(agent)}))
    store = Store(memory=True)
    ws = store.open("reviews", profile=profile)
    ws.files.write("reviews.json", json.dumps(REVIEWS))
    chat = agent.session(ws)
    try:
        out = chat.say(PROMPT)
        print(f"[{out.status}] {out.text}\n")
        if ws.files.exists("summary.json"):
            print("summary.json:", ws.files.read("summary.json").decode())
    finally:
        ws.close()
        store.close()


if __name__ == "__main__":
    main(Agent(os.environ.get("AGEX_MODEL", "anthropic:claude-sonnet-5-5")))
