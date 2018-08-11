import pickle, util

with open("Weights.txt", 'rb') as f:
    W = pickle.load(f)

print W