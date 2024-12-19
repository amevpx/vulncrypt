import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_predictions_and_ground_truth(predictions_file, ground_truth_file):
    """
    Load predictions and ground truth labels from JSON files.
    Args:
        predictions_file (str): Path to the predictions JSON file.
        ground_truth_file (str): Path to the ground truth JSON file.
    Returns:
        Tuple of (predictions, ground_truth).
    """
    with open(predictions_file, 'r') as pred_f:
        predictions = json.load(pred_f)

    with open(ground_truth_file, 'r') as gt_f:
        ground_truth = json.load(gt_f)

    return predictions, ground_truth

def evaluate_model(predictions, ground_truth):
    """
    Evaluate the model's predictions against ground truth labels.
    Args:
        predictions (list of str): List of predicted labels (e.g., ['good', 'bad']).
        ground_truth (list of str): List of actual labels (e.g., ['good', 'bad']).
    """
    assert len(predictions) == len(ground_truth), "Mismatched prediction and ground truth lengths"

    # Convert labels to binary format (e.g., 'good' -> 0, 'bad' -> 1)
    label_map = {'good': 0, 'bad': 1}
    y_pred = [label_map[label] for label in predictions]
    y_true = [label_map[label] for label in ground_truth]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    predictions_file = "predictions.json"  # Path to your predictions file
    ground_truth_file = "ground_truth.json"  # Path to your ground truth file

    print("Loading predictions and ground truth...")
    predictions, ground_truth = load_predictions_and_ground_truth(predictions_file, ground_truth_file)

    # Extract only the labels
    pred_labels = [p['predicted_label'] for p in predictions]
    gt_labels = [gt['label'] for gt in ground_truth]

    print("Evaluating model...")
    evaluate_model(pred_labels, gt_labels)

if __name__ == "__main__":
    main()
