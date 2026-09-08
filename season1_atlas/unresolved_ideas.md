# Season 1 Unresolved Ideas

These ideas are intentionally retained even when the original explanation was weak or overconfident.

## Core Mechanism Hypotheses

- Repeated nonlinear transformation over a small local view can create a useful global representation through state propagation.
- The model can transform partial information immediately instead of waiting for a complete global context representation.
- Early transformations may organize partial information in a way that makes later incoming information easier to use.
- Parameter tying across recurrent hops may make a modest block more powerful per parameter than independently stacked blocks.
- Sparse routing may help by avoiding attention dilution over many irrelevant positions.
- The precise offset geometry may matter less than having a diverse, reachable, structured local view.
- Simple arbitrary, Fibonacci, logarithmic, radix, and harmonic offset systems may all work because they provide multi-scale reachability.

## Routing Questions

- Is learned harmonic routing genuinely better than a fixed offset menu once parameter count and training budget are matched?
- Is the important property spatial structure, multi-scale coverage, or merely repeated message passing?
- Can content-dependent routing be made truly sub-quadratic without materializing an L-by-L salience matrix?
- Does the routing topology need to change at every hop, or is changing the state enough?

## Representation Questions

- Are evolving Q/K/V projections essential, or can identity projections work on some tasks?
- Does V carry the main information while Q/K mainly select locations?
- Is a GRU necessary, or is a wide MLP plus repeated sparse attention sufficient?
- Does the model perform useful intermediate decisions before the final global view is available?
- What exactly propagates across hops: token identity, task state, routing cues, or a mixture?

## Task And Failure Questions

- Why does iterative sparse refinement work so well on some language and vision tasks but struggle with open-vocabulary long-distance copying?
- Does MQAR expose a fundamental sparse-routing limitation or only a poor implementation/training recipe?
- How much of the reported gain comes from the task's local and multi-scale regularities?
- How does performance change when the target dependency is deliberately adversarial to the offset policy?

## Engineering Questions

- What is the smallest useful model and offset count?
- What T gives the best quality/runtime tradeoff?
- Can adaptive halting skip actual work rather than only record a stopping point?
- Which routing implementation is fast on real hardware, not only asymptotically sparse?
- Can one clean implementation support language, associative recall, and vision without separate hand-built architectures?
