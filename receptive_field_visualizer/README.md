# SubQ 3D Receptive Field Explorer

An interactive 3D WebGL visualization built with **Three.js** that demonstrates the core intuition of the SubQ architecture:
> **A token only observes $K$ tokens at a time, but across a small number of recurrent iterations ($T$), its transitive receptive field rapidly cascades to cover the entire sequence.**

---

## How to Run

### Option 1: Python Server (Recommended)
From this directory:
```bash
python serve.py
```
Or from the project root:
```bash
python receptive_field_visualizer/serve.py
```
This will automatically open your default browser at `http://localhost:8080`.

### Option 2: Direct File Open
You can also directly double-click or open `index.html` in any modern web browser (Chrome, Edge, Firefox, Safari).

---

## Features & Controls

1. **Click Any Token in 3D**:
   - Raycasted selection: Click any token along the sequence to observe how the entire context cascades into that specific token.
   - Hover over tokens to see exact token index, relative distance, and offset.

2. **Track Across Iterations ($T$)**:
   - Use the bottom timeline stepper (`[Hop 0]`, `[Hop 1]`, `[Hop 2]`, `[Hop 3]`, `[Hop 4]`).
   - Use `[▶ Play]` for auto-stepping through hops.
   - Watch the telemetry bar update: see the exact receptive field count and sequence coverage percentage scale up!

3. **Two Distinct 3D Perspectives**:
   - **🗼 3D Temporal Cone (Default)**: Visualizes recurrent iterations stacked vertically along the Y-axis. The selected token at the top tier reveals an expanding cone/pyramid of light connecting down to all reached tokens in the raw sequence.
   - **🌈 3D Overhead Arcs**: Tokens sit on a flat runway, and luminous 3D parabolic bridges arch through the sky with heights proportional to distance.

4. **Animated Signals ("Signal Going")**:
   - Luminous energy packets continuously stream along the 3D connection lines/arcs into the observer token.
   - Adjustable signal animation speed slider.

5. **SubQ Multi-Scale vs. Local Window**:
   - **SubQ Multi-Scale (Exponential)**: With $K=5$, a token reaches the entire 96-token sequence in just 3 hops ($K^T = 5^3 = 125 \gg 96$).
   - **Local Window (Linear)**: Notice how standard sliding window attention only inches forward linearly ($T \times K$).
