# Netlib LP Test Problems

Source: https://www.netlib.org/lp/data/
Format: Standard MPS (fixed-width)

| File | Rows | Cols | Optimal | Description |
|------|------|------|---------|-------------|
| afiro.mps | 27 | 32 | -464.7531429 | Smallest Netlib problem |
| kb2.mps | 43 | 41 | -1741.0652703 | Heavy BOUNDS usage |
| sc50a.mps | 50 | 48 | -64.5750771 | Small standard problem |
| sc50b.mps | 50 | 48 | -70.0000000 | Small standard problem |
| blend.mps | 74 | 83 | -30.8121498 | Mixed equality/inequality |

| adlittle.mps | 56 | 97 | 225494.9631 | Mid-scale with BOUNDS |
| share2b.mps | 96 | 79 | -415.7322408 | Sparse mid-scale |
| stocfor1.mps | 117 | 111 | -41131.9763 | Stochastic LP |
| brandy.mps | 220 | 249 | 1518.5098965 | Degenerate (rank-deficient A) |
| scorpion.mps | 388 | 358 | 1878.1248227 | Highly ill-conditioned |
| fit1d.mps | 24 | 1026 | -9146.3780924 | Large-scale (1026 vars), BOUNDS |
| share1b.mps | 117 | 225 | -76589.318579 | Medium degenerate |
| boeing2.mps | 167 | 143 | -315.01872802 | RANGES usage (interval constraints) |

## References

Optimal values obtained from:
- https://www.netlib.org/lp/data/readme
- https://people.clas.ufl.edu/hager/objective-value-comparisons-for-netlib-test-problems/

Files downloaded from:
- afiro.mps: https://github.com/coin-or/CyLP
- kb2.mps, sc50a.mps, sc50b.mps, blend.mps: https://github.com/SkyLiu0/netlib
- brandy.mps, scorpion.mps, fit1d.mps, share1b.mps: https://github.com/SkyLiu0/netlib
- boeing2.mps: https://github.com/coin-or-tools/Data-Netlib (gzip compressed)
