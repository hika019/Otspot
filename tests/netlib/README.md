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

## References

Optimal values obtained from:
- https://www.netlib.org/lp/data/readme
- https://people.clas.ufl.edu/hager/objective-value-comparisons-for-netlib-test-problems/

Files downloaded from:
- afiro.mps: https://github.com/coin-or/CyLP
- kb2.mps, sc50a.mps, sc50b.mps, blend.mps: https://github.com/SkyLiu0/netlib
