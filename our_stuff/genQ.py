from util import Counter
import pickle

Q = Counter()
with open("Q.txt", 'wb') as f:
    pickle.dump(Q, f, protocol=2)