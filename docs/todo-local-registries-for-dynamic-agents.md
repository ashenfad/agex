# Design Doc: Local Registries for Dynamic Agents

## 1. The Problem: Dynamic Agents Don't Survive Process Boundaries

Currently, `agex` supports persistent state through `Versioned(Disk(...))` storage, which works well for statically defined agents. However, dynamically created agents (like in the `dogfood.py` example) break when crossing process boundaries.

**What works today:**
```python
# Static agent definition
my_agent = Agent(name="static-agent", primer="...")
state = Versioned(Disk("/path/to/state"))

# This works across process restarts because the agent is statically defined
result = my_agent_task(data, state=state)
```

**What breaks:**
```python
# Dynamic agent creation (from dogfood.py)
@architect.task
def create_specialist(prompt: str) -> Callable:
    # Creates a new agent dynamically
    with Agent() as specialist:
        # ... configure specialist ...
        return specialist.task(some_function)

# This task_fn breaks when saved to state and loaded in a new process
# because the dynamically created agent isn't in the new process's registry
math_solver = create_specialist("solve math problems")
```

The issue is that dynamically created agents are registered in the global `_AGENT_REGISTRY` of the current process, but this registry doesn't persist. When the `task_fn` is saved to state and loaded in a new process, it can't resolve its agent reference.

## 2. The Dual Registry Problem

Investigation reveals there are actually **two** global registries that cause cross-process issues:

### 2.1 Agent Registry (`_AGENT_REGISTRY`)
- Stores agent instances by fingerprint
- Used to resolve agent references in task functions
- Cleared when process ends

### 2.2 Dynamic Dataclass Registry (`_DYNAMIC_DATACLASS_REGISTRY`)
- Stores dynamically created input dataclasses for task functions
- Required for pickle serialization of dataclass instances
- Also cleared when process ends

**Critical insight:** Dynamic agents create dataclasses using `create_inputs_dataclass_from_ast_args()` which **doesn't register them globally**, while static agents use `_create_inputs_dataclass()` which **does register them globally**. This means dynamic agent task inputs can't be serialized properly.

## 3. The Proposed Solution: Local Registries in Agent State

Instead of relying on global registries, we store the "recipes" for recreating dynamic agents and their dataclasses in the parent agent's persistent state.

### 3.1 Core Concept

The key insight is that from the parent agent's perspective, the "dynamic" agent is actually quite static. The parent creates it with a deterministic set of capabilities based on its own available registry.

```python
# Parent agent's state would contain:
state.set("__local_agent_registry__", {
    "math_solver_123": {
        "primer": "You are a math solver...",
        "capabilities": [
            {"type": "module", "name": "math"},     # Symbolic ref to parent's math module
            {"type": "class", "name": "Agent"}      # Symbolic ref to parent's Agent class
        ],
        "task_definitions": {
            "solve_math_problem": {
                "docstring": "Solve mathematical equations",
                "signature": {...}
            }
        }
    }
})

state.set("__local_dataclass_registry__", {
    "SolveMathProblemInputs": {
        "fields": [("equation", "str")],
        "module": "agex.agent.task"
    }
})
```

### 3.2 How It Works

1. **Agent Creation:** When a parent agent creates a dynamic agent, it stores the agent's "recipe" in its local registry
2. **Task Function Creation:** The returned task function carries both:
   - Parent agent fingerprint (for global registry lookup)
   - Local agent ID (for local registry lookup)
3. **Cross-Process Execution:** When the task function is called in a new process:
   - Resolve parent agent from global registry (it's static)
   - Load parent's state and local registries
   - Reconstruct the dynamic agent using symbolic references
   - Register any needed dataclasses globally
   - Execute the task

### 3.3 Hierarchical Support

The existing `Namespaced` state system naturally supports nested dynamic agents:

```python
# Each level gets its own namespace and local registries
architect_state = Namespaced(base_state, "architect")
math_solver_state = Namespaced(base_state, "architect/math_solver")
equation_parser_state = Namespaced(base_state, "architect/math_solver/equation_parser")
```

This allows agents creating agents creating agents, with each level maintaining its own local registry.

## 4. Implementation Plan

### 4.1 Core Changes

1. **Add Local Registry Storage**
   - Modify `BaseAgent` to manage local registries in state
   - Add methods: `_store_local_agent()`, `_resolve_local_agent()`

2. **Modify Task Creation**
   - Update `TaskMixin.task()` to detect dynamic agent creation context
   - Store agent definition in parent's local registry
   - Modify `TaskUserFunction` to carry local agent reference

3. **Update Task Execution**
   - Modify task function resolution to check local registries first
   - Add dataclass rehydration logic
   - Ensure proper global registration of reconstructed items

4. **Extend Dataclass Handling**
   - Make `create_inputs_dataclass_from_ast_args()` store dataclass definitions
   - Add dataclass reconstruction from stored definitions

### 4.2 Compatibility

This approach:
- **Preserves all existing APIs** - no breaking changes to static agent usage
- **Maintains current behavior** - static agents continue to work exactly as before
- **Additive only** - adds persistence for dynamic agents without affecting anything else

## 5. Benefits

- **Enables persistent dynamic agents:** The `dogfood.py` pattern works across process boundaries
- **Supports nested hierarchies:** Agents creating agents creating agents all work naturally
- **Maintains security model:** Child agents can only access parent capabilities through symbolic references
- **Leverages existing infrastructure:** Built on top of `Namespaced`, `Versioned`, and symbolic reference systems
- **Minimal implementation:** Adds a persistence layer without changing core APIs

## 6. Key Design Properties

- **Isolation:** Each agent level has its own registry namespace, preventing conflicts
- **Automatic cleanup:** Local registries are cleaned up when agent state is garbage collected
- **Environment adaptation:** Symbolic references resolve to whatever implementations are available in the target environment
- **Security preservation:** All existing security boundaries and capabilities inheritance rules are maintained

This design transforms dynamic agents from live, process-bound entities into persistent, portable units that can survive process boundaries. 