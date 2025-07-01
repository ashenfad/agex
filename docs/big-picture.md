# The Big Picture: Agents That Think in Code

This document explains the vision and architectural principles behind `agex`. Unlike frameworks that constrain agents to JSON tool calls, `agex` gives agents a familiar Python environment where they can think, explore, and build solutions using real code.

## Core Philosophy: Code as the Language of Reasoning

The key insight is that **code is language made formal enough to get stuff done**. The same tools that help human developers manage complexity work naturally for AI agents:

- **REPLs** for interactive exploration and step-by-step reasoning
- **`dir()` and `help()`** for discovering capabilities  
- **State inspection** for understanding what's available
- **Modular imports** for accessing functionality
- **Function definitions** for building reusable tools

Instead of inventing new agent interaction patterns, `agex` adapts the proven tools developers have used for decades. This makes agents more effective and their behavior more predictable.

## Implementation Foundation

### REPL-Like Agent Environment

Each agent operates in a persistent, REPL-like environment where they can:
- See recently updated state in their context
- View recent stdout (including errors from previous evaluations)
- Access introspection tools (`help`, `dir`, `print`)
- Build and test solutions iteratively

This environment feels familiar to developers and provides agents with the cognitive scaffolding they need for complex reasoning.

### Dynamic Context Management & Token Budgeting

Context window management uses a sophisticated budgeting system:

- **Token-aware rendering**: The `render` package provides smart `repr` that estimates token costs
- **Dynamic forgetfulness**: Exponentially decaying budget allocation - recent state/stdout gets high fidelity, older information gets compressed
- **Budget allocation**: Configurable token budgets across recent state, stdout, conversation history, and available capabilities
- **Graceful degradation**: When context limits are reached, older information is summarized rather than truncated

This enables agents to maintain relevant context while staying within token limits, without requiring manual context management.

### Validation & Error Handling

**Validation Strategy:**
- Two levels: simple `isinstance` type checks or depth-limited Pydantic validation
- Performance-conscious: validate heads of large lists rather than entire structures
- Immediate feedback: validation failures appear in agent's stdout for the next iteration

**Error Recovery:**
- Validation errors bounce back to the agent with specific failure details
- Agents can see the error and retry within the task loop
- Bounded iteration limits prevent infinite retry loops
- All errors appear in the agent's familiar stdout context

### State Management & Persistence

**Per-Task Behavior:**
- Without `state` parameter: ephemeral environment for single-shot execution
- With `state` parameter: persistent memory across multiple calls
- Recent stdout and state changes appear in agent context for continuity

**Concurrent Agent Isolation:**
- Agents can share the same underlying state through namespacing (`state/namespaced.py`)
- Each agent sees its own isolated view while enabling controlled information sharing
- Clean separation prevents cross-contamination while allowing collaboration

## Runtime Interoperability

### Seamless Python Integration

A key differentiator of this framework is **runtime interoperability** - agents don't just execute code in isolation, they create objects that live and work directly in your Python runtime.

**True Callable Generation:**
```python
@my_coder.task
def make_a_function(prompt: str) -> Callable:
    """Generate a Python function from a text description."""
    pass

# Returns an actual Python callable you can use immediately
prime_finder = make_a_function("Find the next prime larger than a given number")
next_prime = prime_finder(100)  # Works with existing code
my_list.sort(key=prime_finder)  # Integrates with standard library
```

### Beyond Tool Isolation

Most agent frameworks force a choice between:
- **Limited evaluation**: Simple string/JSON processing that can't handle complex Python objects
- **Full isolation**: VM/Docker sandboxing that's secure but completely separate from your runtime

This framework provides a third option: **runtime integration** where agents create real Python objects that participate in your existing codebase.

**Data Processing Handoffs:**
```python
# Seamless data flow between your context and agent context
messy_dataframe = pd.read_csv("complex_data.csv")

@data_agent.task
def clean_and_analyze(df: pd.DataFrame) -> dict:
    """Clean a pandas DataFrame and extract analytical insights."""
    pass

insights = clean_and_analyze(messy_dataframe)
# insights is a real dict in your session - no serialization needed
```

**Dynamic Code Extension:**
```python
# Agent extends your existing classes with new capabilities
@my_coder.task
def add_method_to_class(cls: type, method_description: str) -> type:
    """Dynamically add a new method to an existing class."""
    pass

EnhancedProcessor = add_method_to_class(MyDataProcessor, "add outlier detection")
# Your class now has the new method, usable immediately
```

### Natural Agent Orchestration

Because agents return real Python objects, complex multi-agent workflows become simple Python control flow:

```python
# Generator-critique loop using standard Python
rpt = research_expert("please research ...")
while not (judgement := judge(rpt)).approved:
    rpt = research_revise(judgement.feedback)

# Parallel processing with list comprehensions  
analyses = [specialist_agent(data_chunk) for data_chunk in dataset]

# Conditional branching based on agent outputs
if classifier_agent(document).confidence > 0.8:
    result = expert_agent(document)
else:
    result = human_review_agent(document)
```

No workflow graphs, YAML configurations, or orchestration DSLs needed - just Python.

### Living Codebase Integration

This enables workflows impossible with isolated execution:
- **Collaborative development**: Agents become pair programming partners who directly contribute to shared codebases
- **Dynamic library extension**: Agents add new capabilities to existing systems at runtime
- **Live code evolution**: Functions and classes can be enhanced and optimized without breaking existing interfaces
- **Natural orchestration**: Multi-agent workflows expressed as familiar Python control structures

The result is agents that don't just help *with* your code - they become part of your development environment.

## Hierarchical Agent Architecture

### Agent-to-Agent Communication

Functions can be decorated as both capabilities and tasks:

```python
@orchestrator.fn
@research_expert.task
def deep_research(topic: str) -> ResearchReport:
    """Conduct comprehensive research on the given topic."""
    pass
```

This enables natural hierarchies where specialist agents serve as capabilities for generalist orchestrators.

### Side-Channel Communication

Agents can communicate through a `log()` builtin:

```python
log("Found 12 relevant papers, starting analysis", to="parent_agent")
log("Need human guidance on conflicting sources", to="system")
log("Focusing on theory, suggest you handle applications", to="analysis_agent")
```

Messages appear in the target agent's stdout, fitting naturally into the REPL environment. Role-based targeting provides security boundaries and clear communication channels.

### Shared but Namespaced State

Multiple agents can collaborate while maintaining isolation:
- Shared underlying state with agent-specific namespaces
- Cross-agent communication through explicit channels (function calls, logging)
- Clean separation prevents accidental interference
- Enables complex multi-agent workflows with clear boundaries

## Agent Evolution & Self-Improvement

### On-the-Job Tool Creation

Agents naturally create helper functions while solving tasks. Over time, they accumulate personal toolkits tailored to their common challenges.

### Refactoring & Code Organization

Agents can be given explicit refactoring tasks:

```python
@agent.task("Review and refactor your helper functions for better organization.")
def refactor_my_toolkit() -> None:
    """
    Review and refactor the agent's accumulated helper functions.
    
    Looks for:
    - Repeated patterns that could be abstracted
    - Functions that could be combined or split  
    - Better naming or documentation
    - Opportunities for reusable utilities
    
    Returns:
        None (performs refactoring in place)
    """
    pass
```

This mirrors how human developers evolve their codebases, allowing agents to develop their own programming practices and preferences.

### Module Creation & Interface Design

Agents can formalize their ad-hoc tools into proper modules:
- Choose which functions to make public vs. private
- Create clean interfaces that hide implementation details
- Save context by forgetting internals while remembering signatures/docs
- Share modules with other agents or contribute to global registries

This enables **hierarchical abstraction** - agents build up from primitives to complex tools, then compress implementation details into clean interfaces.

### Emergent Specialization

Different agents develop different approaches:
- Research agents build sophisticated data analysis toolkits
- Writing agents develop tone analysis and structure optimization tools
- Debugging agents create specialized introspection utilities

Agents become more effective over time not just at specific tasks, but at getting better - true **meta-learning**.

## Recursive Agent Architecture (Speculative)

### Dynamic Sub-Agent Creation

The most speculative possibility: agents creating their own specialist sub-agents on demand.

```python
# Agent realizes it needs specialized help
@self.create_subagent("statistics_expert").task("Perform deep statistical analysis with hypothesis testing.")
def run_advanced_analysis(dataset: DataFrame) -> StatisticalReport:
    """
    Run advanced statistical analysis on the dataset.
    
    Args:
        dataset: The DataFrame to analyze
        
    Returns:
        A comprehensive statistical report with hypothesis test results
    """
    pass

# Immediately equip the sub-agent
@statistics_expert.module(scipy.stats, visibility="high")

# Delegate the work
results = run_advanced_analysis(my_data)
```

### Self-Designing Cognitive Architecture

Agents could analyze their own task patterns and architect their cognitive division of labor:
- Micromanager agents creating highly specialized sub-agents
- Architect agents building a few powerful generalist sub-agents
- Collaborative agents creating peer networks
- Hierarchical agents building deep specialization trees

### Recursive Improvement

Sub-agents could create their own sub-agents, leading to:
- Emergent organizational patterns as successful architectures spread
- Agent-driven framework evolution
- Recursive optimization of both code and cognitive structure
- Collaborative architecture design across agent communities

This transforms agents from "users of tools" into "architects of intelligence" - responsible not just for solving problems, but for designing the cognitive structures to solve them effectively.

## Developer Experience & Debugging

### Unified Introspection

The same tools work for both agents and humans:
- `view(agent)` - See available functions/classes/modules
- `view(state)` - Current state and stdout
- `view(conversation)` - Full interaction history  
- `view(hierarchy)` - Agent relationships and communication patterns

### Native Python Introspection

Agent-created functions integrate seamlessly with Python's built-in introspection:

```python
# Agent creates a function
custom_fn = my_agent.create_analyzer("detect data anomalies")

# Standard Python introspection just works
help(custom_fn)
# Output includes rich context:
# - Original user prompt
# - Agent's reasoning process
# - Validation tests performed
# - Dependencies and assumptions

dir(custom_fn)               # Standard attributes
inspect.signature(custom_fn) # Function signature
custom_fn.__doc__            # Enhanced docstring with creation context
```

This makes agent-created code completely native to existing Python tooling - IDEs show rich docstrings on hover, documentation generators include agent context, and debugging tools work as expected.

### Natural Debugging Patterns

Since agents work in familiar Python environments:
- Standard debugging approaches apply
- Error messages and stack traces work as expected
- State inspection uses familiar patterns
- Logging and communication are explicit and traceable
- Agent-created functions are debuggable with standard Python tools

### Collaborative Development

Agents become true programming partners rather than sophisticated tools:
- They participate in the software development process
- Create, refactor, and optimize their own code
- Develop preferences and practices over time
- Share knowledge and tools with other agents and humans

## Security & Boundaries

### Natural Security Points

The registration system provides clean security boundaries:
- Explicit capability registration prevents unauthorized access
- Role-based communication limits cross-agent interference  
- Namespaced state prevents accidental data leakage
- Validation ensures type safety at agent boundaries

### Trust Boundaries

Agents can only call explicitly registered functions, creating clear trust boundaries. The visibility system provides additional control over what capabilities are exposed to which agents.

## Implications & Future Directions

This framework has the potential to transform how we think about agent systems:

### From Tools to Partners

Agents evolve from "smart function callers" to collaborative programmers who:
- Develop their own tools and practices
- Make architectural decisions about their own capabilities
- Share knowledge and techniques with other agents
- Participate in the software development lifecycle

### Emergent Intelligence

The combination of persistent memory, tool creation, refactoring capabilities, and potential recursive architecture could lead to:
- Exponential rather than linear capability growth
- Self-improving agent ecosystems  
- Collaborative agent communities
- Novel organizational patterns and problem-solving approaches

### Natural Adoption

By building on familiar Python patterns and developer mental models, the framework should be uniquely adoptable - developers can apply existing intuitions about programming environments, debugging, and system architecture directly to agent design.

The **runtime interoperability** eliminates the typical friction of agent integration. Instead of restructuring workflows around agent frameworks, agents seamlessly slot into existing codebases as enhanced development tools. Agent-created functions work with standard Python tooling, participate in existing type systems, and integrate with established debugging practices.

The framework doesn't just make agents more capable; it makes them more **collaborative**, more **understandable**, and more **human-compatible** by leveraging the cognitive tools we've already developed for working with complex systems. 