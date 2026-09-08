# Season 2

Season 2 is the clean active research track for iterative sparse local refinement.

Season 1 remains preserved in the parent repository and is indexed by [the Season 1 atlas](../season1_atlas/README.md). New experiments belong here unless they are specifically archival.

## Working Hypothesis

Repeatedly applying a nonlinear transformation to a small structured neighborhood can produce useful global representations through state propagation, while allowing the model to transform partial information immediately.

This is a hypothesis to test, not a claim that the mechanism is already understood.

## Initial Scope

- One parameter-tied physical block.
- Sparse local routing over a configurable offset policy.
- Q/K/V projected from the current state at every hop in the reference implementation.
- A nonlinear MLP update at every hop.
- Explicit causal mode for language and associative-recall tests.
- No GRU, FEN escrow, adaptive halting, diffusion, foundation-model transplant, or custom kernel in the first baseline.

## Research Order

1. Establish a clean local-refinement baseline.
2. Compare fixed arbitrary, dyadic/logarithmic, Fibonacci, and harmonic routing.
3. Measure recurrent depth and quality/runtime tradeoffs.
4. Run MQAR to test exact identity propagation.
5. Run TinyShakespeare for causal language modeling.
6. Run CIFAR-100 for cross-domain transfer.
7. Reintroduce GRU, FEN, content routing, halting, and kernel work one at a time.

## Result Rule

Every experiment must save a machine-readable result containing its configuration, parameter count, seed, data split, training budget, runtime, hardware, metrics, and source commit.
