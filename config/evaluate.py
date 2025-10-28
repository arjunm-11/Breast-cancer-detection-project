import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def build_test_generator(test_dir, img_size=(128,128), gray=True, batch_size=32):
    color_mode = 'grayscale' if gray else 'rgb'
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return gen

def plot_confusion_matrix(cm, classes, out_path):
    import seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_pred_proba, classes, out_path):
    # Works for binary; for multi-class extend to one-vs-rest
    if y_pred_proba.shape[1] != 2:
        return
    fpr, tpr, _ = roc_curve(y_true[:,1], y_pred_proba[:,1])
    auc = roc_auc_score(y_true[:,1], y_pred_proba[:,1])
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def main(args):
    os.makedirs('results', exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Building test generator from: {args.test_dir}")
    gen = build_test_generator(
        args.test_dir,
        img_size=(args.img_size[0], args.img_size[1]),
        gray=args.gray,
        batch_size=args.batch_size
    )

    # Predict
    print("Running inference...")
    y_prob = model.predict(gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = gen.classes
    class_indices = gen.class_indices
    idx_to_class = {v:k for k,v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Metrics
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification report:\n", report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # Save artifacts
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    np.savetxt('results/confusion_matrix.csv', cm, fmt='%d', delimiter=',')

    # ROC (binary case)
    y_true_onehot = np.eye(len(class_names))[y_true]
    try:
        plot_roc(y_true_onehot, y_prob, class_names, 'results/roc_curve.png')
    except Exception as e:
        print(f"ROC plot skipped: {e}")

    # Confusion matrix plot
    try:
        plot_confusion_matrix(cm, class_names, 'results/confusion_matrix.png')
    except Exception as e:
        print(f"CM plot skipped: {e}")

    # Overall accuracy and loss (optional evaluate)
    loss, acc = model.evaluate(gen, verbose=0)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/baseline_cancernet.h5", help="Path to .h5 model")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Test directory with class subfolders")
    parser.add_argument("--img_size", type=int, nargs=2, default=[128,128], help="Image size H W")
    parser.add_argument("--gray", action="store_true", help="Use grayscale mode")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
