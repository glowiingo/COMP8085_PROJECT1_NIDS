from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Temp test running
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

attack_cat_str_values = []

class dtc:
    def __init__(self, df: pd.DataFrame, x_train, x_test, x_val, label_train, label_test, label_val, attack_cat_train, attack_cat_test, attack_cat_val):
        self.df = df
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.label_train = label_train
        self.label_test = label_test
        self.label_val = label_val
        self.attack_cat_train = attack_cat_train
        self.attack_cat_test = attack_cat_test
        self.attack_cat_val = attack_cat_val
        attack_cat_str_values = df['attack_cat'].unique()
        attack_cat_str_values[0] = "None"
    
    # EXPERIMENT PART 1 - RUN CLASSIFIER WITHOUT SELECTED FEATURES

    def no_selected_features_label(self):
        print("Calculating Accuracy Score without using Selected Features for Label Prediction....")
        clf = DecisionTreeClassifier(criterion='entropy')
        clf_label = clf.fit(self.x_train, self.label_train)
        # save this into a pickle file
        label_pred = clf_label.predict(self.x_test)
        acc_score = metrics.accuracy_score(self.label_test, label_pred)*100
        print("Label Prediction Accuracy on Decision Tree Classifier with No Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.label_test, label_pred))

    def no_selected_featues_attack_cat(self):
        print("Calculating Accuracy Score without using Selected Features for Attack Category Prediction....")
        clf = DecisionTreeClassifier(criterion='entropy')
        clf_attack_cat = clf.fit(self.x_train, self.attack_cat_train)
        # save into pickle file
        attack_pred = clf_attack_cat.predict(self.x_test)
        acc_score = metrics.accuracy_score(self.attack_cat_test, attack_pred)*100
        print("Attack Category Prediction Accuracy on Decision Tree Classifier with No Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.attack_cat_test, attack_pred, target_names=attack_cat_str_values))

    def 