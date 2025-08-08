# Local Registries for Dynamic Agents

Persist recipes for dynamically created agents so their callable tasks survive process restarts.

## Concept
- Store symbolic recipes in parent agent state (primer, capabilities, task signatures).
- Reconstruct agents cross-process using Namespaced state and allowlists.

## Flow
1. Task carries parent fingerprint + local agent ID.
2. Resolve parent from global registry; load state.
3. Rebuild dynamic agent from recipe; register needed dataclasses.
4. Execute task with the reconstructed agent.

## Benefits
- Durable dogfood pattern; unlimited nesting.
- Works across restarts and distributed deployments.
- Preserves security via symbolic capability references.

## Considerations
- Stable IDs, garbage collection, schema migration.
- No raw object leakage; enforce allowlisted reconstruction.

Related issue: [Issue #4](https://github.com/ashenfad/agex/issues/4)
