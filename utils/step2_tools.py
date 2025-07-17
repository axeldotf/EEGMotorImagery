import os
import numpy as np
import matplotlib.pyplot as plt


def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray, n_subjects: int | None = None) -> None:
    """
    Plot and save the class distribution for training and testing datasets.

    Parameters
    ----------
    y_train : array-like
        Labels for the training set (binary: 0=Rest, 1=Imagery).
    y_test : array-like
        Labels for the testing set (binary: 0=Rest, 1=Imagery).
    n_subjects : int or None, optional
        Number of subjects to include in the filename and title (default: None).

    Saves
    -----
    PNG image of the class distribution bar plot in 'output_img/class_distribution' folder.
    """
    labels = ['Rest (0)', 'Imagery (1)']

    # Count occurrences of each class in train and test sets
    train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    test_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]

    x = range(len(labels))
    # Plot bars for training and testing sets side by side
    bar_train = plt.bar(x, train_counts, width=0.4, label='Train', align='center')
    bar_test = plt.bar([i + 0.4 for i in x], test_counts, width=0.4, label='Test', align='center')

    # Configure x-axis labels and plot title
    plt.xticks([i + 0.2 for i in x], labels)
    plt.ylabel("Number of samples")
    title_suffix = f" (# Subjects: {n_subjects})" if n_subjects is not None else ""
    plt.title(f"Class Distribution: Train vs Test{title_suffix}")
    plt.legend()

    # Annotate bars with counts above them
    max_count = max(train_counts + test_counts)
    for bar in bar_train + bar_test:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_count * 0.01 + 1,
            str(height),
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Prepare output directory
    output_dir = os.path.join("output_img", "class_distribution")
    os.makedirs(output_dir, exist_ok=True)

    # Compose filename
    filename = f"class_distribution_{n_subjects}S.png" if n_subjects is not None else "class_distribution.png"
    save_path = os.path.join(output_dir, filename)

    # Save figure and display
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"✅ Class distribution plot saved as '{save_path}'")
