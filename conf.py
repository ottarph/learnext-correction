OutputLoc = "../LearnExt/Output" # Specifiy which relative location problem data is stored in
train_checkpoints = range(0, 1800) # Specify which checkpoints are to be regarded as training data.
test_checkpoints = range(1800, 2400+1) # Specify which checkpoints are to be regarded as test data.
vandermonde_loc = "models/vandermonde.t" # Location of saved vandermonde matrices, compute offline to save time.
