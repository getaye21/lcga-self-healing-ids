"""
LCGA (Lightweight CNN-GRU-Attention) Model Architecture
~48K parameters — designed for real-time network intrusion detection.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np, os, time

def build_lcga(n_features, n_classes):
    inp = tf.keras.Input(shape=(n_features, 1), name="input")

    # Block 1: Dual-scale parallel CNN branches
    branch_a = layers.Conv1D(32, 3, padding="same", name="conv_a")(inp)
    branch_a = layers.BatchNormalization()(branch_a)
    branch_a = layers.Activation("relu")(branch_a)
    branch_a = layers.Dropout(0.2)(branch_a)
    branch_a = layers.MaxPooling1D(2, padding="same")(branch_a)

    branch_b = layers.Conv1D(64, 5, padding="same", name="conv_b")(inp)
    branch_b = layers.BatchNormalization()(branch_b)
    branch_b = layers.Activation("relu")(branch_b)
    branch_b = layers.MaxPooling1D(2, padding="same")(branch_b)

    x = layers.Concatenate(name="concat")([branch_a, branch_b])

    # Block 2: GRU for temporal modelling
    x = layers.GRU(64, return_sequences=True, name="gru")(x)
    x = layers.Dropout(0.2)(x)

    # Block 3: Multi-Head Self-Attention + Residual
    x = layers.Dense(32, name="proj")(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16, name="mha")(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Block 4: Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    return tf.keras.Model(inp, out, name="LCGA")


class LCGATrainer:
    def __init__(self, n_features, n_classes, model_dir):
        self.n_features = n_features
        self.n_classes = n_classes
        self.model_dir = model_dir
        self.model = None
        self.history = None

    def build(self):
        self.model = build_lcga(self.n_features, self.n_classes)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, X_tr, y_tr, X_val, y_val, epochs=60, batch_size=256):
        cb_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss"),
            callbacks.ModelCheckpoint(os.path.join(self.model_dir, "lcga_best.keras"), save_best_only=True),
        ]
        self.history = self.model.fit(
            X_tr, y_tr, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=cb_list, verbose=1
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_te, y_te):
        probs = self.predict(X_te)
        y_pred = np.argmax(probs, axis=1)
        from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
        return {
            "accuracy": accuracy_score(y_te, y_pred),
            "macro_f1": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_te, y_pred, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_te, y_pred),
        }

    def measure_latency(self, X_sample, n_runs=500):
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.model(X_sample, training=False)
            times.append((time.perf_counter() - t0) * 1000)
        return {"mean_ms": np.mean(times[50:]), "std_ms": np.std(times[50:])}

    def save(self, name="lcga_cicids.keras"):
        self.model.save(os.path.join(self.model_dir, name))
