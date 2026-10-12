"""A session: an agent working in a workspace, turn by turn.

    AGEX_MODEL=anthropic:claude-sonnet-5-5 python examples/session.py

The workspace keeps the agent's files and its conversation on one
branch, so the second turn remembers the first, and a checkout of an
earlier commit would rewind both.
"""

import os

from nontainer import Store

from agex import Agent


def main(agent: Agent) -> None:
    store = Store(memory=True)
    ws = store.open("demo")
    chat = agent.session(ws)
    try:
        out = chat.say(
            "Write fib.py printing the first ten Fibonacci numbers, and run it."
        )
        print(f"[{out.status}] {out.text}\n")

        out = chat.say("Make it print twenty instead, and run it again.")
        print(f"[{out.status}] {out.text}\n")

        print("fib.py now reads:")
        print(ws.files.read("fib.py").decode())
        print(f"{len(chat.runs)} runs, {len(ws.log())} commits")
    finally:
        ws.close()
        store.close()


if __name__ == "__main__":
    main(Agent(os.environ.get("AGEX_MODEL", "anthropic:claude-sonnet-5-5")))
