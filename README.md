# LearnExt: NN-corrected-training

Code repository for the training problem of the NN-corrected harmonic extension in manuscript
> J. Haubner, O. Hellan, M. Zeinhofer, M. Kuchta: Learning Mesh Motion Techniques with Application to Fluid-Structure Interaction, arXiv preprint arXiv:2206.02217

Any code that needs ``fem_nets`` must install [from here](https://github.com/MiroK/fem-nets).


Otherwise, requirements are generally
- `PyTorch`
- `FEniCS` (legacy)
- `numpy`
- `matplotlib`

Scripts to run a specific learning problem are located in the `problems/`-folder. Code here is imported from other locations.

To use without ``MeshView``-functionality, you can install with conda from ``environment.yml`` and then ``pip``-install ``fem_nets`` from local repository.
