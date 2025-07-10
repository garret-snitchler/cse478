import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def load_preprocess_data(val_split=0.1):
    print("Loading CIFAR100 data")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    val_size = int(len(x_train) * val_split)
    print(f"Validation size: {val_size}")
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train2 = x_train[val_size:]
    y_train2 = y_train[val_size:]

    print("Train shape:", x_train2.shape)
    print("Val shape: ", x_val.shape)
    print("test shape:", x_test.shape)

    return (x_train2, y_train2), (x_val, y_val), (x_test, y_test)

def plot_history(history, title=''):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title(f'Accuracy {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    #loss
    plt
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'Loss {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_conf_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation='vertical', values_format='d')
    plt.show()



print("\nTesting load_and_preprocess_data:")
(train_x, train_y), (val_x, val_y), (test_x, test_y) = load_preprocess_data()
print(f"Train shape: {train_x.shape}, Labels: {train_y.shape}")
print(f"Val shape: {val_x.shape}, Labels: {val_y.shape}")
print(f"Test shape: {test_x.shape}, Labels: {test_y.shape}")