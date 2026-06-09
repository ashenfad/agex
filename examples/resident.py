"""
The Resident Agent

One long-lived agent identity, two ways in:

  - a cron loop that triages a (simulated) inbox every few seconds
  - an interactive chat where you talk to the *same* agent

Both run in ONE process against ONE kvgit branch ("home") — the agent's
durable identity. A custom StateResolver gives each entry point its own
working tree over that shared branch: sessions "cron" and "chat" each
resolve to an independent `Staged`, so neither blocks the other, and
overlapping commits reconcile through kvgit's CAS + three-way merge
(the task loop's `on_conflict` handling) instead of last-write-wins.

The agent's durable self is its files, not any one conversation:

  - policies/triage.md   — how to handle email; editable from chat
  - drafts/<id>.md       — replies it wasn't allowed to send
  - cache["claim/<id>"]  — idempotency claims so overlapping triage
                           runs never double-handle an email

`send_reply` is gated behind scope="email" — locked by default, so the
agent drafts instead of sending. Grant it live from chat with /grant.
Policy changes land the same way: ask chat to edit policies/triage.md
and the next cron tick obeys (each turn refreshes its working tree, so
cross-channel changes appear at the next turn boundary).

Run everything in one process:

    python resident.py

Or split across terminals (same substrate, same semantics):

    python resident.py cron
    python resident.py chat

Try: asking what's in the inbox, then "tighten the invoice policy:
flag anything over $100", then watch the next cron tick obey. Then
/grant email and watch drafts become sends.
"""

import json
import os
import sys
import tempfile
import threading
import time

from agex import Agent, connect_llm, connect_state, scopes
from agex.state import assert_safe_session, commit_state, staged_state
from agex.state.kv import Disk

# ---------------------------------------------------------------------------
# Shared substrate: one store, one branch, a working tree per entry point
# ---------------------------------------------------------------------------

HOME = os.path.join(tempfile.gettempdir(), "agex-resident")
BRANCH = "home"
INBOX_PATH = os.path.join(HOME, "inbox.json")
OUTBOX_PATH = os.path.join(HOME, "outbox.json")

os.makedirs(HOME, exist_ok=True)
_store = Disk(os.path.join(HOME, "state"))


class HomeResolver:
    """Every session is its own working tree over the shared "home" branch.

    Session "cron" and session "chat" get independent `Staged` instances
    pinned to the same branch: one writer per tree, optimistic CAS +
    three-way merge between trees. (The built-in storages can't express
    this — they'd give each session a separate substrate entirely.)
    """

    versioned = True

    def __init__(self):
        self._cache: dict = {}

    def resolve(self, session: str):
        assert_safe_session(session)
        if session not in self._cache:
            self._cache[session] = staged_state(_store, branch=BRANCH)
        return self._cache[session]


llm = connect_llm(provider="anthropic", model="claude-haiku-4-5")

resident = Agent(
    name="resident",
    primer=(
        "You are a resident assistant with a persistent workspace. "
        "Your standing instructions live in policies/triage.md — read it "
        "before triaging. Your memory is your files; keep them tidy."
    ),
    llm=llm,
    state=connect_state(type="resolver", resolver=HomeResolver()),
    isolation="none",
)


def _refresh(session: str) -> None:
    """Pull the branch's latest commit into this session's working tree.

    Each channel commits independently; refreshing at the turn boundary is
    what lets chat see cron's latest triage (and vice versa) without
    waiting for a commit-time merge.
    """
    staged = resident.state(session)
    if not staged.has_changes:  # type: ignore[union-attr]
        staged.refresh()  # type: ignore[union-attr]


# The (simulated) outside world: send_reply has real side effects, so it
# is scope-gated. Until the scope is granted, calls raise ScopeRequired
# and the agent falls back to drafting (per its task primer).


@resident.fn(scope="email")
def send_reply(to: str, subject: str, body: str) -> str:
    """Send an email reply. Requires the 'email' scope."""
    outbox = _read_json(OUTBOX_PATH, [])
    outbox.append({"to": to, "subject": subject, "body": body, "at": time.time()})
    _write_json(OUTBOX_PATH, outbox)
    return f"sent to {to}"


# ---------------------------------------------------------------------------
# Tasks: two entry points into one identity
# ---------------------------------------------------------------------------


@resident.task(
    primer=(
        "Triage these unread emails per policies/triage.md. For each email:\n"
        "1. If cache.get('claim/' + email['id']) is set, another run already "
        "has it — skip it and do NOT include it in your handled list.\n"
        "2. Otherwise set cache['claim/' + email['id']] = True before acting.\n"
        "3. Reply via send_reply() if policy allows. If send_reply raises a "
        "scope error, write the reply to drafts/<id>.md instead — do not "
        "request permission; an unattended run has no one to ask.\n"
        "4. Note anything policy says to flag in memory/alerts.md (append).\n"
        "Return the ids you actually handled (claimed) this run."
    ),
    on_conflict="abandon",  # a dropped tick redoes naturally next tick;
    # abandoned commits also roll back claims, so no email is stranded
)
def triage(emails: list[dict]) -> list[str]:  # type: ignore[empty-body]
    """Triage unread emails; return the ids handled this run."""
    pass


@resident.task(
    primer=(
        "You are chatting with your principal. You share state with your "
        "own background email-triage runs — their activity is in your event "
        "log, their output is in your files. When asked to change how email "
        "is handled, edit policies/triage.md directly; the next triage run "
        "will follow it. Show diffs of policy edits you make."
    ),
    on_conflict="retry",  # foreground turn: re-run on collision
)
def chat(message: str) -> str:  # type: ignore[empty-body]
    """Converse with the resident agent."""
    pass


# ---------------------------------------------------------------------------
# Host plumbing: fake inbox, json helpers, seeding
# ---------------------------------------------------------------------------

FAKE_ARRIVALS = [
    {
        "id": "m1",
        "from": "ann@example.com",
        "subject": "Lunch Thursday?",
        "body": "Want to grab lunch Thursday at noon?",
    },
    {
        "id": "m2",
        "from": "billing@vendor.example",
        "subject": "Invoice #4471",
        "body": "Your invoice for $180 is attached. Due in 14 days.",
    },
    {
        "id": "m3",
        "from": "sam@example.com",
        "subject": "Quick question",
        "body": "What timezone are you in these days?",
    },
    {
        "id": "m4",
        "from": "billing@vendor.example",
        "subject": "Invoice #4502",
        "body": "Monthly subscription invoice: $49.",
    },
]

DEFAULT_POLICY = """\
# Triage policy

- Meeting/social requests: draft (or send, if allowed) a brief, warm reply
  accepting tentatively; mention I'll confirm by end of day.
- Questions with factual answers I know from my files: answer them.
- Invoices: do not reply. Flag invoices over $150 in memory/alerts.md.
- Anything unclear: leave it unread for my principal.
"""


def _read_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: str, value) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def seed() -> None:
    fs = resident.fs("chat")
    if not fs.exists("policies/triage.md"):
        fs.write("policies/triage.md", DEFAULT_POLICY.encode())
        # fs writes only stage; commit so the cron tree sees the policy
        commit_state(resident.state("chat"))  # type: ignore[arg-type]
    if not os.path.exists(INBOX_PATH):
        _write_json(INBOX_PATH, {"pending": list(FAKE_ARRIVALS), "unread": []})


# ---------------------------------------------------------------------------
# Cron channel — deliver one pending email per tick, then triage unread
# ---------------------------------------------------------------------------


def run_cron(tick_seconds: float = 20.0) -> None:
    seed()
    while True:
        inbox = _read_json(INBOX_PATH, {"pending": [], "unread": []})
        if inbox["pending"]:
            arrived = inbox["pending"].pop(0)
            inbox["unread"].append(arrived)
            print(f"\n[cron] arrived: {arrived['subject']!r}")
        if inbox["unread"]:
            _refresh("cron")  # see chat's latest policy edits and grants
            handled = triage(list(inbox["unread"]), session="cron")
            if handled is None:  # abandoned on conflict; next tick redoes
                print("[cron] tick abandoned (concurrent commit); will retry")
            else:
                inbox["unread"] = [m for m in inbox["unread"] if m["id"] not in handled]
                print(f"[cron] handled: {handled or 'nothing'}")
        _write_json(INBOX_PATH, inbox)
        time.sleep(tick_seconds)


# ---------------------------------------------------------------------------
# Chat channel — talk to the same agent; /grant and /revoke flip scopes
# ---------------------------------------------------------------------------


def run_chat() -> None:
    seed()
    print("Chatting with the resident. /grant email, /revoke email, /quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line.startswith(("/grant", "/revoke")):
            verb, _, scope = line.partition(" ")
            s = scopes(resident.state("chat"))
            (s.grant if verb == "/grant" else s.revoke)(scope.strip() or "email")
            print(f"[host] {verb[1:]}ed scope; cron picks it up next tick")
            continue
        _refresh("chat")  # see cron's latest triage activity
        print(f"resident> {chat(line, session='chat')}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "cron":
        run_cron()
    elif mode == "chat":
        run_chat()
    elif mode == "both":
        # One process, two channels, one branch: the resolver gives each
        # session an independent working tree, so the cron thread and the
        # chat REPL never share a staging area.
        threading.Thread(target=run_cron, daemon=True).start()
        run_chat()
    else:
        print(__doc__)
