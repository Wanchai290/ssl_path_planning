# SSL Path Planning

This repo has implementation of RRT,
RRT* and Informed RRT* algorithms for path planning.
Algorithms are stored in the `algs` module, and execution
lays in the `main.py` file. You can change which algorithm is used
and then run from the repo's root directory using :

```bash
py main.py
```

## References
1. S.Karaman, E.Frazzoli, Sampling-based Algorithms for Optimal Motion Planning 
https://arxiv.org/pdf/1105.1186

2. S.M. LaValle, Rapidly-Exploring Random Trees: A New Tool for Path Planning
https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf

3. Daichi Miyajima et al., KIKS Extended Team Description
for RoboCup 2023, https://ssl.robocup.org/wp-content/uploads/2023/02/2023_ETDP_KIKS.pdf