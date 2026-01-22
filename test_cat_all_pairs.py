import pickle

num_list = [5000, 10000, 15000, 23781]

all_pair_with_intrinsics = []
for n in num_list:
    with open(f'test_dataset_output/pairs_{n}_with_intrinsics.pkl', "rb") as f:
        pair_with_intrinsics = pickle.load(f)
    all_pair_with_intrinsics.extend(pair_with_intrinsics)
print(len(all_pair_with_intrinsics))
with open(f'test_dataset_output/all_pairs_test_with_intrinsics.pkl', "wb") as f:
    pickle.dump(all_pair_with_intrinsics, f)