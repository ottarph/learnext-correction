# learnext-learning-problem

Networks and architectures for learnExt problem of mesh extension.

The ``fem_nets``-portions need to install [from here](https://github.com/MiroK/fem-nets) for them to work.

Otherwise, requirements are generally
- `PyTorch`
- `FEniCS` (legacy)
- `numpy`
- `matplotlib`

Scripts to run a specific learning problem are located in the `problems/`-folder. Code here is imported from other locations.

To use without ``MeshView``-functionality, you can install with conda from ``environment.yml`` and then ``pip``-install ``fem_nets`` from local repository.
