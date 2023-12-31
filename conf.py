OutputLoc = "../LearnExt/Output" # Specifiy which relative location problem data is stored in
mesh_file_loc = OutputLoc + "/Mesh_Generation/"
data_file_loc = OutputLoc + "/Extension/Data/"
harmonic_file_loc = data_file_loc + "input_.xdmf"
biharmonic_file_loc = data_file_loc + "output_.xdmf"

harmonic_label = "input_harmonic_ext"
biharmonic_label = "output_biharmonic_ext"

vandermonde_loc = "models/vandermonde.t" # Location of saved vandermonde matrices, compute offline to save time.

train_checkpoints = range(0, 1800)          # Specify which checkpoints are to be regarded as training data.
validation_checkpoints = range(1800, 2000)  # Specify which checkpoints are to be regarded as validation data.
test_checkpoints = range(2000, 2400+1)      # Specify which checkpoints are to be regarded as test data.

poisson_mask_f = "2.0 * (x[0]+1.0) * (1-x[0]) * exp( -3.5*pow(x[0], 7) ) + 0.1"

with_submesh = True
submesh_conversion_loc = "tools/conversion_arrays/"
submesh_conversion_cg1_loc = submesh_conversion_loc + "mv_to_sm_cg1.npy"
submesh_conversion_cg2_loc = submesh_conversion_loc + "mv_to_sm_cg2.npy"
