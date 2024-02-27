from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from enum import Enum
from Constants import *
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics.pairwise import np

class knn:
    def __init__(self):
        self.knn_neighbors_label = 21
        self.knn_neighbors_attack_cat = 21
        self.knn_label = None
        self.knn_attack_cat = None
        self.scaler = StandardScaler()
        self.classifier_label = 'Classifier: K Nearest Neighbors'

    def train_label_model(self, X, y_label):
        X_label_train = X[SELECTED_FEATURES_LABEL_RFE]
        X_label_train = self.scaler.fit_transform(X_label_train)
        knn_label = KNeighborsClassifier(n_neighbors=self.knn_neighbors_label)    
        self.knn_label = knn_label.fit(X_label_train, y_label)



    def train_attack_cat_model(self, X, y_attack_cat):
        X_attack_cat_train = X[SELECTED_FEATURES_ATTACK_CAT_RFE]
        X_attack_cat_train = self.scaler.fit_transform(X_attack_cat_train)
        knn_attack_cat = KNeighborsClassifier(n_neighbors=self.knn_neighbors_attack_cat)
        self.knn_attack_cat = knn_attack_cat.fit(X_attack_cat_train, y_attack_cat)
        
    

    def pickle_label_model(self):      
        with open('knn_label.pkl', 'wb') as f:
            pickle.dump(self.knn_label, f)
    
    def pickle_attack_cat_model(self):
        with open('knn_attack.pkl', 'wb') as f:
            pickle.dump(self.knn_attack_cat, f)
    

    def print_prediction_report_for_label(self, X, y):        
        X_label_test = X[SELECTED_FEATURES_LABEL_RFE]
        X_label_test = self.scaler.fit_transform(X_label_test)
        y_label_pred = self.knn_label.predict(X_label_test)
        print(self.classifier_label)
        print(metrics.classification_report(y, y_label_pred))

    def print_prediction_report_for_attack_cat(self, X, y):
        X_attack_cat_test = X[SELECTED_FEATURES_ATTACK_CAT_RFE]
        X_attack_cat_test= self.scaler.fit_transform(X_attack_cat_test)
        y_pred_attack_cat = self.knn_attack_cat.predict(X_attack_cat_test)
        print(self.classifier_label)
        print(metrics.classification_report(y, y_pred_attack_cat))

    def get_label_model(self):
        return self.knn_label
    
    def get_attack_cat_model(self):
        return self.knn_attack_cat
    
    def perform_validation_attack_cat(self, x_train, y_train, x_val, y_val):
        acc_vals = []
        x_train_cat = x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]
        x_train_cat = self.scaler.fit_transform(x_train_cat)
        x_val_cat = x_val[SELECTED_FEATURES_ATTACK_CAT_RFE]
        x_val_cat = self.scaler.fit_transform(x_val_cat)
        
        k_values = [i for i in range (1,31)]
        for k in k_values:
            knnc = KNeighborsClassifier(n_neighbors=k)
            knnc.fit(x_train_cat, y_train)
            y_pred = knnc.predict(x_val_cat)
            
            accuracy = accuracy_score(y_val, y_pred)
            print('K value: ' + str(k) + ' Accuracy: ' + str(accuracy))
            acc_vals.append(accuracy)
        sns.lineplot(x = k_values, y = acc_vals, marker = 'o')
        plt.title('Accuracy of Different K Values for Attack Category')
        plt.xlabel("K Values")
        plt.ylabel("Accuracy Score")
        plt.show()
        
    def perform_validation_label(self, x_train, y_train, x_val, y_val):
        acc_vals = []
        x_train_cat = x_train[SELECTED_FEATURES_LABEL_RFE]
        x_train_cat = self.scaler.fit_transform(x_train_cat)
        x_val_cat = x_val[SELECTED_FEATURES_LABEL_RFE]
        x_val_cat = self.scaler.fit_transform(x_val_cat)
        
        k_values = [i for i in range (1,31)]
        for k in k_values:
            knnc = KNeighborsClassifier(n_neighbors=k)
            knnc.fit(x_train_cat, y_train)
            y_pred = knnc.predict(x_val_cat)
            
            accuracy = accuracy_score(y_val, y_pred)
            print('K value: ' + str(k) + ' Accuracy: ' + str(accuracy))
            acc_vals.append(accuracy)
        sns.lineplot(x = k_values, y = acc_vals, marker = 'o')
        plt.title('Accuracy of Different K Values for Label')
        plt.xlabel("K Values")
        plt.ylabel("Accuracy Score")
        plt.show()