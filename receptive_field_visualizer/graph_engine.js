/**
 * Graph Engine for SubQ Receptive Field Computation
 * Computes receptive fields across recurrent iterations (hops).
 */

class ReceptiveFieldGraph {
  constructor(options = {}) {
    this.seqLen = options.seqLen || 96;
    this.K = options.K || 5;
    this.mode = options.mode || 'multiscale'; // 'multiscale' or 'local'
    this.isCausal = options.isCausal || false; // bidirectional by default for clear visualization
    this.maxHops = options.maxHops || 5;
  }

  /**
   * Get direct offsets for a given hop
   * In multiscale mode, stride expands as K^(hop - 1), yielding K^T combinatorial reach!
   * In local mode, stride is always 1, yielding standard linear expansion.
   */
  getOffsets(hop = 1) {
    const K = this.K;
    const stride = (this.mode === 'multiscale') ? Math.pow(K, hop - 1) : 1;
    const offsets = [];

    if (this.isCausal) {
      // Causal: look only at current and past tokens (i - off)
      for (let s = 0; s < K; s++) {
        let off = s * stride;
        if (off >= this.seqLen) {
          off = Math.min(this.seqLen - 1, s * Math.max(1, Math.floor(this.seqLen / K)));
        }
        offsets.push(off);
      }
    } else {
      // Bidirectional: symmetric around 0
      const radius = Math.floor(K / 2);
      for (let r = -radius; r <= radius; r++) {
        let off = r * stride;
        const maxOff = Math.floor(this.seqLen / 2);
        if (Math.abs(off) >= maxOff) {
          off = Math.sign(off) * Math.min(maxOff, Math.abs(r) * Math.max(1, Math.floor(maxOff / Math.max(1, radius))));
        }
        offsets.push(off);
      }
      if (offsets.length > K) {
        offsets.pop();
      } else if (offsets.length < K) {
        offsets.push((radius + 1) * stride);
      }
    }

    return offsets.sort((a, b) => a - b);
  }

  /**
   * Get direct neighbor tokens that token `idx` looks at in a given hop
   */
  getDirectNeighbors(idx, hop = 1) {
    const offsets = this.getOffsets(hop);
    const neighbors = new Set();
    for (const off of offsets) {
      const target = this.isCausal ? (idx - off) : (idx + off);
      if (target >= 0 && target < this.seqLen) {
        neighbors.add(target);
      }
    }
    return Array.from(neighbors).sort((a, b) => a - b);
  }

  /**
   * Computes the exact backwards transitive cone of influence
   * from targetIdx at tier currentHop down to tier 0 (raw input sequence).
   */
  computeCone(targetIdx, currentHop) {
    const tierActive = {};
    for (let h = 0; h <= currentHop; h++) {
      tierActive[h] = new Set();
    }
    tierActive[currentHop].add(targetIdx);

    const edges = [];
    const edgeSet = new Set();

    for (let h = currentHop; h >= 1; h--) {
      for (const node of tierActive[h]) {
        const neighbors = this.getDirectNeighbors(node, h);
        for (const nbr of neighbors) {
          tierActive[h - 1].add(nbr);
          const edgeKey = `${nbr}@${h - 1}->${node}@${h}`;
          if (!edgeSet.has(edgeKey)) {
            edgeSet.add(edgeKey);
            edges.push({
              from: nbr,
              to: node,
              fromTier: h - 1,
              toTier: h,
              hop: h
            });
          }
        }
      }
    }

    const tierArrays = {};
    for (let h = 0; h <= currentHop; h++) {
      tierArrays[h] = Array.from(tierActive[h]).sort((a, b) => a - b);
    }

    const rawCovered = tierArrays[0] && tierArrays[0].length > 0 ? tierArrays[0] : [targetIdx];
    const minIdx = Math.min(...rawCovered);
    const maxIdx = Math.max(...rawCovered);

    return {
      currentHop,
      targetIdx,
      tierActive: tierArrays,
      rawCovered,
      edges,
      receptiveFieldCount: rawCovered.length,
      coveragePct: (rawCovered.length / this.seqLen) * 100,
      span: maxIdx - minIdx + 1,
      minIdx,
      maxIdx
    };
  }
}

// Export for browser
window.ReceptiveFieldGraph = ReceptiveFieldGraph;
