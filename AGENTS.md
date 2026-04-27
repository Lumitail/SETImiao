
# ACS Agent Guide

Work issue-by-issue.
Always preserve the `.dat` contract:
- rows x 256
- 2 bytes per cell
- packed signed-int8 I/Q
- metadata external
- FFT along time within each coarse channel

Before changing code:
1. state the intended outcome,
2. state the acceptance test,
3. change code,
4. run tests,
5. run the real 2-second smoke sample for major pipeline changes.
