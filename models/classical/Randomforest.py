from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False

    def flatten(self, X):
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train, y_train):
        print(" Mokome Random Forest modelį...")
        X_train_flat = self.flatten(X_train)

        # (Pasirinktinai) naudoti tik dalį duomenų greitesniam mokymui
        sample_size = min(5000, len(X_train_flat))
        self.model.fit(X_train_flat[:sample_size], y_train[:sample_size])
        self.is_fitted = True
        print(" Modelis apmokytas.")

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Modelis dar nebuvo apmokytas.")
        X_flat = self.flatten(X)
        return self.model.predict(X_flat)

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise ValueError("Modelis dar nebuvo apmokytas.")
        print(" Vertiname Random Forest modelį...")
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n Tikslumas: {acc * 100:.2f}%")
        print("\n Klasifikacijos ataskaita:")
        print(classification_report(y_test, y_pred))
        print("\n Klaidų matrica:")
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        return acc

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Random Forest klaidų matrica')
        plt.xlabel('Prognozuota klasė')
        plt.ylabel('Tikroji klasė')
        plt.tight_layout()
        plt.show()