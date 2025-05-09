import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import joblib


def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Churn_Yes'])
    y = df['Churn_Yes']
    return X, y


def train_svm(X_train, y_train):
    model = SVC(random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"\nSupport Vector Machine Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\nSupport Vector Machine Test Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")


def main():
    X, y = load_data('../../data/processed/telco_customer_churn_encoded.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_svm(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, '../../models/support_vector_machine_model.pkl')


if __name__ == "__main__":
    main()
