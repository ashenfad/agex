# The Interpreter Project: A Path to True Streaming

## The Goal: Solving Hierarchical Streaming

The current AST visitor-based evaluation system (`agex/eval`) has a fundamental architectural limitation: it is a synchronous, depth-first recursive system. This prevents true, real-time event streaming in two key scenarios:

1.  **Hierarchical Agents**: When a parent agent calls a sub-agent, the parent's `visit_Call` method blocks until the sub-agent has run to **complete completion**. This causes all of the sub-agent's events (`TaskStart`, `Action`, `Output`, etc.) to be delivered in a single batch after a long pause, defeating the purpose of real-time streaming.
2.  **Built-in Outputs**: Similarly, when an agent's code calls `print()` or `view_image()`, the `OutputEvent` can only be yielded *after* the entire code block evaluation is finished, not at the moment the call is made.

The goal of the Interpreter Project is to replace the `eval` system with a new execution engine that solves this problem, enabling a fluid, real-time, hierarchical streaming experience.

## Proposed Architecture: A Custom Interpreter

The proposed solution is to build a new interpreter from the ground up in a parallel `agex/interp` module. This architecture consists of two main stages: a **Compiler** that transforms code into a custom instruction sequence, and a **Virtual Machine (VM)** that executes it.

This approach gives us explicit control over the execution loop and our own **call stack**, which is the key to solving the streaming problem.

### Key Components in `agex/interp`

1.  **Compiler (`compiler.py`)**:
    *   Uses the `ast.NodeVisitor` pattern to walk the source code's AST one time.
    *   Its job is **not** to execute code, but to emit a linear sequence of custom `Instruction` objects (our "bytecode").
    *   For example, `a + b` would be compiled into a sequence like: `[LOAD_NAME 'a', LOAD_NAME 'b', BINARY_ADD]`.

2.  **Instructions & Opcodes (`instructions.py`, `opcodes.py`)**:
    *   Defines the custom instruction set for our VM (e.g., `LOAD_CONSTANT`, `STORE_NAME`, `CALL_FUNCTION`, `JUMP_IF_FALSE`).
    *   An `Instruction` is a simple data object containing an `opcode` and an optional `argument`.

3.  **Virtual Machine (`vm.py`)**:
    *   The core execution engine. It is a single, flat `while` loop that runs as long as there are frames on the call stack.
    *   It reads the next `Instruction` and modifies its internal state (operand stacks, program counters) accordingly.
    *   This is the "state machine" that cycles on its own state, completely separate from the event stream it emits.

4.  **Call Frame (`frame.py`)**:
    *   A data structure that represents a single function call. The VM will manage a `call_stack` of these frames.
    *   Crucially, this object holds the execution state for its scope.

### Integration with `agex/state`

The new interpreter will **not** reinvent scope management. It will be built directly on top of the powerful, composable `agex/state` system.

*   **`CallFrame.scope`**: Instead of a simple `locals` dictionary, each `CallFrame` will contain a single `scope` attribute, which will be an `agex.state.State` object (typically an instance of `Scoped`).
*   **Opcodes Use the State API**: Opcodes like `LOAD_NAME` and `STORE_NAME` will call `current_frame.scope.get(name)` and `current_frame.scope.set(name, value)`. The `Scoped` state object will automatically handle traversing the parent scope chain.
*   **Closures**: When the compiler encounters a `def` statement, it will create a `Closure` object that captures its defining environment by holding a reference to the **current frame's `scope` object**.
*   **Function Calls**: The `CALL_FUNCTION` opcode will create a new `CallFrame` for the callee. It will instantiate a new `Scoped` state for this frame, setting its `parent_store` to the `scope` captured by the `Closure`. This elegantly replicates Python's lexical scoping using the existing, proven infrastructure.

### How This Solves the Streaming Problem

This architecture cleanly separates the interpreter's internal control flow from the external event stream.

1.  **The VM is in Control**: The main `while` loop of the VM drives all execution.
2.  **Streaming is an Emission**: When an instruction for a streaming operation is encountered (e.g., a call to `print` or a sub-agent), the VM's handler for that instruction can simply `yield` the relevant event (`OutputEvent`, etc.).
3.  **Execution Continues**: After yielding, the `while` loop simply continues to its next cycle and executes the next instruction. The `yield` does not affect the VM's internal call stack or control flow.
4.  **Hierarchical Streaming**: When `CALL_FUNCTION` targets a sub-agent, the VM can pause the execution of the current frame, consume the entire sub-agent's event stream (yielding each event as it arrives), and then resume the parent frame's execution once the sub-agent is complete.

### Phased Implementation Plan

This is a significant undertaking that should be approached systematically.

1.  **Phase 1: Define the Machine**: Create the core data structures in `agex/interp`: `opcodes.py`, `instructions.py`, and the `state`-integrated `frame.py`.
2.  **Phase 2: Build the Compiler**: Use `ast.NodeVisitor` in `compiler.py` to translate Python features into instructions, starting with the simplest expressions and incrementally adding statements, control flow (`if`/`for`/`try`), and function definitions.
3.  **Phase 3: Build the VM**: Concurrently, implement the logic for each opcode inside the VM's main loop in `vm.py`. The VM's complexity will grow with the compiler's capabilities.
4.  **Phase 4: Test-Driven Development**: Create a parallel test suite in `tests/agex/interp` by copying and modifying the existing `eval` tests. This allows for validating parts of the interpreter (e.g., expressions, then loops) independently.
5.  **Phase 5: Integration**: Once the interpreter has feature-parity and the new test suite is passing, update the agent's task loop to use the new engine. Afterwards, the old `agex/eval` module can be safely removed.

### Conclusion: A Strategic Investment

Building a custom interpreter is a **multi-week project** requiring focused effort. It represents a trade-off between delivering other features in the short term and paying down significant technical debt for a superior long-term architecture. The current partial-streaming solution is functional, but this project outlines the path to achieving the ideal streaming behavior that the `agex` framework was designed for. 