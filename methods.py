from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import os


class ML_Models:
    def __init__(self, base_path):
        self.model_path = base_path
        self.initial_models = []
        self.models = {}
        self.initial_models.append(('LogisticRegression', LogisticRegression()))
        self.initial_models.append(('DecisionTree', DecisionTreeClassifier()))
        self.initial_models.append(('RandomForest', RandomForestClassifier()))
        self.initial_models.append(('NaiveBayes', MultinomialNB()))
        self.initial_models.append(('KNN', KNeighborsClassifier()))
        self.initial_models.append(('SVM', SVC()))

    def set_data(self, X, y, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit_models(self):
        models = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        for name, model in self.initial_models:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            models.append(name)
            accuracy.append(round(accuracy_score(self.y_test, y_pred), 4) * 100)
            precision.append(round(precision_score(self.y_test, y_pred), 4) * 100)
            recall.append(round(recall_score(self.y_test, y_pred), 4) * 100)
            f1.append(round(f1_score(self.y_test, y_pred), 4) * 100)
            pickle.dump(model, open(os.path.join(self.model_path, name + '.pkl'), 'wb'))
            self.models[name] = model
        self.outputs = pd.DataFrame({'Model': models, 'Accuracy': accuracy, 'Precision': precision,
                                     'Recall': recall, 'F1 Score': f1})
        self.outputs = self.outputs.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    def get_outputs(self):
        return self.outputs

    def run(self, X, y):
        self.set_data(X, y)
        self.fit_models()
        print("ML Models Outputs: \n", self.get_outputs())

    def delete_initial_model(self, model_name):
        self.initial_models = [model for model in self.initial_models if model[0] != model_name]

    def load_model(self, model_name):
        model = pickle.load(open(os.path.join(self.model_path, model_name + '.pkl'), 'rb'))
        self.models[model_name] = model

    def predict(self, model_name, X):
        return self.models[model_name].predict(X)