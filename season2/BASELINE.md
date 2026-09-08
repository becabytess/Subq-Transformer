# Season 2 Baseline Specification

## Name

`IterativeLocalSubQ`

The name describes the mechanism without assuming that harmonic waves are fundamental.

## Computation

Given token states `s^(0)` and a candidate offset policy `D`, repeat for `t = 1..T`:

1. Normalize the current state.
2. Project Q, K, and V from the current state.
3. Gather only the candidates selected by `D` for each query position.
4. Compute sparse attention over those candidates.
5. Apply the output projection and nonlinear MLP update.
6. Add the update to the current state.

The parameters are shared across hops. The reference version is causal when `D` contains only self and backward offsets and when candidate validity is enforced explicitly.

## Routing Policies

The model must expose one common routing interface for:

- Arbitrary fixed offsets.
- Dyadic/logarithmic offsets.
- Fibonacci offsets.
- Learned harmonic offsets.

The first comparison should use identical model width, MLP width, hop count, optimizer, batches, seeds, and evaluation protocol.

## Deliberate Omissions

The first baseline excludes GRU gates, FEN escrow, content-salience top-k over all positions, adaptive halting, diffusion decoding, and foundation-model weight surgery. These can be added later as isolated variants.

## Primary Questions

- Does repeated nonlinear local refinement outperform a one-layer dense model at equal parameter count?
- How much does quality improve as T increases?
- Does the result hold on exact associative recall, not only language statistics?
- Which offset policy provides the best quality/runtime tradeoff?

## Baselines

- One-layer dense Transformer with matched parameters.
- Four-layer dense Transformer as a capacity/depth reference.
- IterativeLocalSubQ with T=1, 2, 4, 8.
- IterativeLocalSubQ with each routing policy.
