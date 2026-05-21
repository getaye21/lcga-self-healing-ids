"""DT Surrogate + SHAP + LIME explainability module."""
import shap, lime, lime.lime_tabular
import numpy as np, joblib, time, os
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

class DTSurrogate:
    def __init__(self, feature_names, class_names, model_dir):
        self.feature_names = feature_names
        self.class_names = class_names
        self.model_dir = model_dir
        self.dt = None
        self.shap_explainer = None
        self.lime_explainer = None

    def fit(self, X, y_lcga_hard):
        self.dt = DecisionTreeClassifier(max_depth=8, criterion="entropy", min_samples_leaf=5, random_state=42)
        self.dt.fit(X, y_lcga_hard)

    def fidelity(self, X, y_lcga_hard):
        return np.mean(self.dt.predict(X) == y_lcga_hard)

    def build_shap(self):
        self.shap_explainer = shap.TreeExplainer(self.dt)

    def build_lime(self, X_bg):
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_bg, feature_names=self.feature_names, class_names=self.class_names,
            mode="classification", random_state=42)

    def global_importance(self, X, save_path=None):
        if not self.shap_explainer: self.build_shap()
        sv = self.shap_explainer.shap_values(X[:500])
        shap.summary_plot(sv, X[:500], feature_names=self.feature_names, show=False)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return plt.gcf()

    def shap_vs_lime_timing(self, X, predict_fn):
        n = len(X)
        t0 = time.perf_counter()
        self.shap_explainer.shap_values(X)
        shap_ms = (time.perf_counter() - t0) / n * 1000
        t0 = time.perf_counter()
        for xi in X:
            self.lime_explainer.explain_instance(xi, predict_fn, num_features=5)
        lime_ms = (time.perf_counter() - t0) / n * 1000
        return {"shap_ms": shap_ms, "lime_ms": lime_ms}

    def extract_rule(self, x):
        return export_text(self.dt, feature_names=self.feature_names)

    def save(self):
        joblib.dump(self.dt, os.path.join(self.model_dir, "dt_surrogate.pkl"))
