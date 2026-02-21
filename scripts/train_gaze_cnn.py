from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_DIR = Path("dataset")
MODEL_OUT = Path("models/gaze_cnn.h5")
CLASSES = ["focus", "left", "right", "updown"]

def load_data():
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        folder = DATASET_DIR / cls
        for p in folder.glob("*.png"):
            img = tf.keras.utils.load_img(p, color_mode="grayscale", target_size=(64,64))
            arr = tf.keras.utils.img_to_array(img) / 255.0
            X.append(arr)
            y.append(idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def build_model():
    m = models.Sequential([
        layers.Input(shape=(64, 64, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main():
    X, y = load_data()
    if len(X) == 0:
        raise SystemExit("No training images found. Run preprocess_columbia.py first.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    
    cb = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]
    
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, callbacks=cb, verbose=1)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    
    print(f"Saved model to {MODEL_OUT}")
    print(f"Test accuracy: {acc:.4f}")
    
    # Additional evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))

if __name__ == "__main__":
    main()