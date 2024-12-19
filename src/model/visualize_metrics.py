import json
import matplotlib.pyplot as plt


def plot_metrics(metrics_file="training_metrics.json"):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    val_accuracies = metrics["val_accuracies"]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 7))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_metrics()
