"""Binary detection ensemble (ANN + GRU) with stacking meta-learner."""
import numpy as np, joblib, os
from sklearn.linear_model import LogisticRegression
from tensorflow.keras import layers, models, callbacks

class BinaryEnsemble:
    def __init__(self, n_features, seq_shape, model_dir):
        self.n_features = n_features
        self.seq_shape = seq_shape
        self.model_dir = model_dir
        self.ann = None
        self.gru = None
        self.meta = LogisticRegression(solver="liblinear")

    def build_ann(self):
        model = models.Sequential([
            layers.Dense(128, activation="relu", input_shape=(self.n_features,)),
            layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ], name="ANN_Detector")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def build_gru(self):
        model = models.Sequential([
            layers.GRU(64, input_shape=self.seq_shape),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ], name="GRU_Detector")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, X_flat, X_seq, y, X_val_flat, X_val_seq, y_val, epochs=50):
        self.ann = self.build_ann()
        self.gru = self.build_gru()
        cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.ann.fit(X_flat, y, validation_data=(X_val_flat, y_val), epochs=epochs, batch_size=256, callbacks=[cb], verbose=1)
        self.gru.fit(X_seq, y, validation_data=(X_val_seq, y_val), epochs=epochs, batch_size=256, callbacks=[cb], verbose=1)
        # Meta-learner
        p_ann = self.ann.predict(X_val_flat, verbose=0).flatten()
        p_gru = self.gru.predict(X_val_seq, verbose=0).flatten()
        self.meta.fit(np.column_stack([p_ann, p_gru]), y_val)

    def predict(self, X_flat, X_seq):
        p_ann = self.ann.predict(X_flat, verbose=0).flatten()
        p_gru = self.gru.predict(X_seq, verbose=0).flatten()
        return self.meta.predict(np.column_stack([p_ann, p_gru]))

    def predict_proba(self, X_flat, X_seq):
        p_ann = self.ann.predict(X_flat, verbose=0).flatten()
        p_gru = self.gru.predict(X_seq, verbose=0).flatten()
        return self.meta.predict_proba(np.column_stack([p_ann, p_gru]))[:, 1]

    def save(self):
        if self.ann: self.ann.save(os.path.join(self.model_dir, "ann_nsl.keras"))
        if self.gru: self.gru.save(os.path.join(self.model_dir, "gru_nsl.keras"))
        joblib.dump(self.meta, os.path.join(self.model_dir, "meta_nsl.pkl"))
