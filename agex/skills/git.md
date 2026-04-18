---
name: git
description: Version control for your workspace files — checkpoint, branch, diff, and reset.
---

# Git — workspace version control

You have access to a `git` command in the terminal that tracks your workspace files. All file writes (via `<FILE>` or `<TERMINAL>`) are automatically tracked — there is no staging area and no `git add` step.

## Quick reference

### Checkpointing
```bash
git commit -m "describe what you just did"
git log --oneline
```
Commit early and often. Each commit is a named checkpoint you can return to.

### Inspecting changes
```bash
git diff                    # diff HEAD vs previous commit
git diff HEAD~2             # diff HEAD vs 2 commits ago
git show HEAD:path/to/file  # view a file at a specific commit
git status                  # current branch and recent commits
```

### Branching for experiments
```bash
git checkout -b experiment   # create and switch to a new branch
# ... try something ...
git commit -m "attempted approach A"

git checkout main            # switch back
git checkout -b experiment2  # try another approach
# ... try something else ...
git commit -m "attempted approach B"

# keep the one that worked:
git checkout main
git merge experiment2
git branch -d experiment     # delete the failed branch
```

### Recovering from mistakes
```bash
git log --oneline            # find the commit you want
git reset --hard HEAD~1      # undo the last commit
git diff HEAD~1              # check what changed before resetting
```

## Key differences from real git

- **No `git add`** — all file writes are automatically tracked.
- **`git commit -m "msg"`** checkpoints the current state with your message. Every commit must include `-m`.
- **Local only** — no `push`, `pull`, `fetch`, or `remote`. Your workspace is the only copy.
- **No `.git` directory** — git state is managed internally, not as files in your workspace.

## When to use git

- **Before risky changes**: `git commit -m "working state before refactor"`
- **After completing a logical unit of work**: `git commit -m "implemented date parser"`
- **When exploring alternatives**: create a branch, try it, merge or delete
- **When debugging**: `git diff HEAD~1` to see what you just changed
