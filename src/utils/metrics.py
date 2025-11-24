from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def print_metrics(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("F1-score:", f1_score(y_true, y_pred, average='macro'))
    print("ROC AUC:", roc_auc_score(y_true, y_pred, multi_class='ovo'))
