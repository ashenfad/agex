# TODO: Contextual, Capability-Driven Primers

This document outlines a proposed system for dynamically injecting primer instructions into an agent's context based on the capabilities it has been granted.

## 1. Core Philosophy

This system extends the core `agex` philosophy of providing a "guided compute environment." Currently, guidance is primarily offered manually through the `visibility` parameter in registration methods. This proposal adds an automatic, framework-driven layer of guidance.

The principle is: **The framework should intelligently coach the agent on how to use its tools, but only when those tools are relevant.**

This creates a two-pronged system for adaptive agent priming:
-   **Developer-Guided (Manual):** The developer uses `visibility` to control the granularity of information presented to the agent.
-   **Framework-Guided (Automatic):** The framework uses knowledge of its own features and constraints to provide context-sensitive instructions.

## 2. Proposed System Architecture

1.  **Modular Primer Library:** Instead of a single, monolithic primer, the framework would maintain a library of small, single-purpose primer files (e.g., `with_statements.md`, `view_image.md`, `core_exit_fns.md`).

2.  **Registry of Primer Triggers:** A mechanism to associate specific conditions with primer snippets. These triggers would be evaluated when capabilities are registered with an agent.

3.  **Primer Assembler:** A component that runs before each LLM call. It inspects the agent's capabilities, checks the trigger registry, and assembles a final, dynamic primer by combining the necessary snippets with the user-provided primer.

## 3. Initial Use Cases

### a. Handling Stateful / Unpickleable Objects

-   **Trigger:** A call to `agent.module(some_instance, ...)` where `some_instance` is detected as a class instance and not a module.
-   **Action:** Automatically inject a primer that explains:
    1.  The object is stateful and cannot be assigned to a variable when using `Versioned` state.
    2.  The requirement to chain methods (e.g., `db.execute(...).fetchall()`).
    3.  The use of the `with instance as ...:` pattern for safe transactions or temporary operations.
-   **Impact:** This would make user-written primers like `db_primer.py` obsolete, dramatically lowering the barrier to entry for using live objects.

### b. Vision Capabilities (`view_image`)

-   **Trigger:** The agent's capabilities include a function or task that returns an "image-friendly" type (e.g., `plotly.Figure`, `PIL.Image`, `matplotlib.Figure`). This would be detected by inspecting the return type annotations of registered functions and tasks.
-   **Action:** Automatically inject the `view_image` primer, explaining how the agent can see, evaluate, and refine visual output.
-   **Impact:** This improves context efficiency by not showing vision instructions to agents that can't create images. It also reduces the risk of agent confusion or hallucination.

## 4. Future Potential

Once the core system is in place, it can be extended to provide even more sophisticated guidance:

-   **Cost/Performance Coaching:** Inject primers advising caution when an agent is given access to a function marked as "expensive."
-   **Deprecation Warnings:** Automatically warn an agent when it's using a deprecated function and suggest the alternative.
-   **Security Best Practices:** Provide targeted advice for modules with known security considerations (e.g., advising against ReDoS attacks when the `re` module is registered). 