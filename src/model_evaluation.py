from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)

def evaluate_model(y_true, y_pred, model_name="Model"):
    print(f"\n=== {model_name} Evaluation ===")
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')