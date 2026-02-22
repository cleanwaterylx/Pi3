import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

if __name__ == "__main__":
    # Load the data from the .npy file
    data = np.load('./gts_preds_visym_test_pi3_visymscenes_classification_multi_level_feature.npy', allow_pickle=True).item()
    # data = np.load('../doppelgangers-plusplus/eval_visym_ap0.9917_auc0.9902_prec85_1.0000_recall95_0.9119.npy', allow_pickle=True).item()
    # print(data)
    
    # Extract predictions and ground truths
    preds = np.array(data['preds'])
    gts = np.array(data['gts'])
    
    # Calculate True Positives, False Positives, True Negatives, and False Negatives
    tp = np.sum((preds >= 0.5) & (gts == 1))
    fp = np.sum((preds >= 0.5) & (gts == 0))
    tn = np.sum((preds < 0.5) & (gts == 0))
    fn = np.sum((preds < 0.5) & (gts == 1))
    
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

    
    # Calculate AUC and AP
    auc_score = roc_auc_score(gts, preds)
    ap_score = average_precision_score(gts, preds)
    
    
    
    print(f"Average Precision: {ap_score:.4f}")
    print(f"AUC: {auc_score:.4f}")
    
    precision, recall, thresholds = precision_recall_curve(gts, preds)

    from scipy.interpolate import interp1d

    f = interp1d(recall[::-1], precision[::-1])
    prec_at_recall = float(f(0.85))
    print("Prec@Recall>=0.85:", prec_at_recall)

    # Recall@Prec>=0.99
    f = interp1d(precision, recall)
    recall_at_prec = float(f(0.99))
    print("Recall@Prec>=0.99:", recall_at_prec)
    
    # draw roc curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve   
    fpr, tpr, thresholds = roc_curve(gts, preds)
    plt.plot(fpr, tpr, label='ROC curve (area = {:.4f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])   
    plt.xlabel('False Positive Rate')

    