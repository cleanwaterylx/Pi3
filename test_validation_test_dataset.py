import pickle

all_pairs = []
with open("all_pairs_test.pkl", "rb") as f:
    all_pairs = pickle.load(f)

print(len(all_pairs))


threshold = 1e+5
negative_tp = 0
negative_fp = 0
positive_fn = 0
positive_tn = 0
for idx, pair in enumerate(all_pairs):
    if pair[2] == '0':
        flag = pair[3][-1]
        if flag == 0:
            negative_tp += 1
        else:
            negative_fp += 1
    else:
        flag = pair[3][-1]
        if flag == 1:
            positive_tn += 1
        else:
            positive_fn += 1

print(f'Negative pairs Precision: {negative_tp} / {negative_tp + negative_fp}', negative_tp / (negative_tp + negative_fp))
print(f'Positive pairs Precision: {positive_tn} / {positive_tn + positive_fn}', positive_tn / (positive_tn + positive_fn))
print(f'Negative pairs Recall: {negative_tp} / {negative_tp + positive_fn}', negative_tp / (negative_tp + positive_fn))
print(f'Positive pairs Recall: {positive_tn} / {positive_tn + negative_fp}', positive_tn / (positive_tn + negative_fp))
