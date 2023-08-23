
# This script should perform all the steps necessary to create the dataset in a compatible sequence. You can also do it faster in parallel by calling 
# all the scripts yourself and sending them to the background using &, eg.
#   PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop1.py &
#   PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop2.py &
#   PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop3.py &
#   ...,
# making sure to do the steps in a correct order. Assumed to be run from the repository base directory.
#
# Note: The extensions for each of the six load configurations are stored in separate files. 
# When converted to numpy, each snapshot is stored in a separate file, but the numbering is 
# common over all the load configurations.

# Create mesh of the solid and fluid domains
PYTHONPATH=. python3 data_prep/artificial/make_mesh.py

# Compute the solid deformations with scaling cos(theta), theta in np.linspace(0, 2*np.pi, 101) for all six load configurations
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop1.py
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop2.py
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop3.py
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop4.py
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop5.py
PYTHONPATH=. python3 data_prep/artificial/beam_simulations/explore_loop6.py

# Compute the harmonic and biharmonic extensions to the fluid domain of the beam deformations. Note: these are interpolated into CG1 in the scripts, but this is easy to change if needed.
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset1.py
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset2.py
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset3.py
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset4.py
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset5.py
PYTHONPATH=. python3 data_prep/artificial/extension_computations/make_dataset6.py

# Convert to numpy arrays containing vertex displacements, to use with PyTorch
PYTHONPATH=. python3 data_prep/artificial/convert_checkpoints.py

# Test
PYTHONPATH=. python3 data_prep/artificial/test/test_art_dataset.py
