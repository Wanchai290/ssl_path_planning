# SSL Path Planning

This repo has implementation of RRT,
RRT* and Informed RRT* algorithms for path planning.
Algorithms are stored in the `algs` module, and execution
lays in the `main.py` file. You can change which algorithm is used
and then run from the repo's root directory using :

```bash
py main.py
```

## Implementation details
In [1], the cost function is defined such that `Cost(p) = Parent(p) + c(Line(Parent(p), p))`.
Here, cost of a node `p` is merely the current distance traveled up until attained `p`.

The implementation of `CostTreeGraph#get_cost` does not perform this. Instead, the cost of a node
`p` is directly set in the algorithms. This way, it is not required to recursively compute the
cost of a node, thus allowing for O(1) access.

## References
1. S.Karaman, E.Frazzoli, Sampling-based Algorithms for Optimal Motion Planning 
https://arxiv.org/pdf/1105.1186

2. S.M. LaValle, Rapidly-Exploring Random Trees: A New Tool for Path Planning
https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf

3. Daichi Miyajima et al., KIKS Extended Team Description
for RoboCup 2023, https://ssl.robocup.org/wp-content/uploads/2023/02/2023_ETDP_KIKS.pdf