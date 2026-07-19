import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes=["Edible", "Poisonous"]):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    return fig
