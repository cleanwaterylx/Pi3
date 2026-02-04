import pickle

num_list = [5000, 10000, 15000, 23849]

all_pair_with_intrinsics = []
for n in num_list:
    with open(f'pairs_{n}.pkl', "rb") as f:
        pair_with_intrinsics = pickle.load(f)
    all_pair_with_intrinsics.extend(pair_with_intrinsics)
print(len(all_pair_with_intrinsics))
with open(f'all_pairs_test.pkl', "wb") as f:
    pickle.dump(all_pair_with_intrinsics, f)