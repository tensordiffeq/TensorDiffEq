# TensorDiffEq - Efficient Multi-GPU PINN Solvers

[![Build Status](https://travis-ci.com/levimcclenny/TensorDiffEq.svg?token=v7YRnTJ5sKUDc2sKNqG5&branch=main)](https://travis-ci.com/levimcclenny/TensorDiffEq)

Scientific Machine Learning on top of Tensorflow for multi-worker distributed computing. 

Use TensorDiffEq if you require:
- A meshless PINN solver that can distribute over multiple workers (GPUs) for
  forward problems (inference) and inverse problems (discovery)
- Self-Adaptive Collocation methods for forward and inverse PINNs
- Intuitive user interface allowing for explicit definitions of variable domains, 
  boundary contitions, initial conditions, and strong-form PDEs to solve

What makes TensorDiffEq different?
- [Self-Adaptive Solvers](https://arxiv.org/abs/2009.04544) for forward and inverse problems, leading to increased accuracy of the solution
- Multi-GPU distributed training for large or fine problems domains
- built on top of Tensorflow 2.0 for increased support in new functionality exclusive to recent TF releases, such as [XLA support](https://www.tensorflow.org/xla), 
[autograph](https://blog.tensorflow.org/2018/07/autograph-converts-python-into-tensorflow-graphs.html) for efficent graph-building, and [grappler support](https://www.tensorflow.org/guide/graph_optimization)
  for graph optimization*



*In development


