from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from Constants import *
import pickle

# Temp test running
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
        print(metrics.classification_report(self.attack_cat_test, attack_pred, target_names=ATTACK_CAT_STR_VALUES))

    # EXPERIMENT PART 1 - RUN CLASSIFIER WTIH SELECTED FEATURES
    def selected_features_label(self):
        print("Calculating Accuracy Score with Selected Features for Label...")
        clf = DecisionTreeClassifier(criterion='entropy')
        x_train_selected_label = self.x_train[SELECTED_FEATURES_LABEL_RFE]
        x_test_selected_label = self.x_test[SELECTED_FEATURES_LABEL_RFE]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        label_pred = clf_label.predict(x_test_selected_label)
        acc_score = metrics.accuracy_score(self.label_test, label_pred)*100
        print("Label Prediction Accuracy of RFE Selected Features: {:.2f}%".format(acc_score))
        print(metrics.classification_report(self.label_test, label_pred))

    def selected_features_attack(self):
        print("Calculating Accuracy Score with Selected Features for Attack Category...")
        # print("================ RFE SELECTED FEATURES ===============")
        clf = DecisionTreeClassifier(criterion='entropy')
        x_train_selected_attack = self.x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]
        x_test_selected_attack = self.x_test[SELECTED_FEATURES_ATTACK_CAT_RFE]
        clf_attack = clf.fit(x_train_selected_attack, self.attack_cat_train)
        attack_pred = clf_attack.predict(x_test_selected_attack)
        acc_score = metrics.accuracy_score(self.attack_cat_test, attack_pred)*100
        print("Attack Category Prediction Accuracy of RFE Selected Features: {:.2f}%".format(acc_score))
        print(metrics.classification_report(self.attack_cat_test, attack_pred, target_names=ATTACK_CAT_STR_VALUES))
        
        # print("================= EBFI SELECTED FEATURES ===================")
        # clf = DecisionTreeClassifier(criterion='entropy')
        # x_train_selected_attack = self.x_train[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        # x_test_selected_attack = self.x_test[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        # clf_attack = clf.fit(x_train_selected_attack, self.attack_cat_train)
        # attack_pred = clf_attack.predict(x_test_selected_attack)
        # acc_score = metrics.accuracy_score(self.attack_cat_test, attack_pred)*100
        # print("Attack Category Prediction Accuracy of EBFI Selected Features: {:.2f}%".format(acc_score))
        # print(metrics.classification_report(self.attack_cat_test, attack_pred, target_names=ATTACK_CAT_STR_VALUES))

    # PART 2 - Optimized the training and classifier so that best possible scores are retrieved for Labels
    def optimal_training_selected_features_label(self):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth = 9)
        x_train_selected_label = self.x_train[SELECTED_FEATURES_LABEL_RFE]
        x_val_selected_label = self.x_val[SELECTED_FEATURES_LABEL_RFE]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        label_pred = clf_label.predict(x_val_selected_label)
        classifier_name = "Decision Tree Classifier"
        print("\nClassifier: {}\n".format(classifier_name))
        print(metrics.classification_report(self.label_val, label_pred))

    # PART 3 - Optimized the training and classifier so that best possible scores are retrieved for Attack Category
    def optimal_training_selected_features_attack(self):
        # print("================ RFE SELECTED FEATURES ===============")
        clf = DecisionTreeClassifier(criterion='entropy', max_depth = 16)
        x_train_selected_attack_cat = self.x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]
        x_val_selected_attack_cat = self.x_val[SELECTED_FEATURES_ATTACK_CAT_RFE]
        clf_attack_cat = clf.fit(x_train_selected_attack_cat, self.attack_cat_train)
        attack_cat_pred = clf_attack_cat.predict(x_val_selected_attack_cat)
        classifier_name = "Decision Tree Classifier"
        print("\nClassifier: {}\n".format(classifier_name))
        print(metrics.classification_report(self.attack_cat_val, attack_cat_pred, target_names=ATTACK_CAT_STR_VALUES))

        # print("================= EBFI SELECTED FEATURES ===================")
        # clf = DecisionTreeClassifier(criterion='entropy', max_depth = 19)
        # x_train_selected_attack_cat = self.x_train[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        # x_val_selected_attack_cat = self.x_val[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        # clf_attack_cat = clf.fit(x_train_selected_attack_cat, self.attack_cat_train)
        # attack_cat_pred = clf_attack_cat.predict(x_val_selected_attack_cat)
        # classifier_name = "Decision Tree Classifier"
        # print("\nClassifier: {}\n".format(classifier_name))
        # print(metrics.classification_report(self.attack_cat_val, attack_cat_pred, target_names=ATTACK_CAT_STR_VALUES))

    def get_experiment_data_part_one(self):
        print("============= PART ONE EXPERIMENTS =============")
        self.no_selected_features_label()
        self.no_selected_featues_attack_cat()
        self.selected_features_label()
        self.selected_features_attack()
        print("============= END OF PART ONE EXPERIMENTS =============")


if __name__ == '__main__':
    df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True)
    df = df.replace(r'\s+', '', regex=True)
    # df.fillna('None', inplace=True)
    df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)

    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].astype('str')
    df['is_ftp_login'] = df['is_ftp_login'].astype('str')
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].astype('str')

    df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
    df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")

    # converting str to int
    df['attack_cat'] = pd.factorize(df['attack_cat'])[0]
    df['proto'] = pd.factorize(df['proto'])[0]
    df['state'] = pd.factorize(df['state'])[0]
    df['service'] = pd.factorize(df['service'])[0]

    df['ct_flw_http_mthd'] = pd.factorize(df['ct_flw_http_mthd'])[0]
    df['is_ftp_login'] = pd.factorize(df['is_ftp_login'])[0]
    df['ct_ftp_cmd'] = pd.factorize(df['ct_ftp_cmd'])[0]

    df['srcip'] = preprocessing.LabelEncoder().fit_transform(df['srcip'])
    df['dstip'] = preprocessing.LabelEncoder().fit_transform(df['dstip'])
    feature_cols = list(df.columns[0:-2]) # remove label and attack_cat

    X_features = df[feature_cols] # Features
    label = df['Label'] # Variable
    attack_cat = df['attack_cat'] # Variable
    X_train, X_temp, attack_cat_train, attack_cat_temp, label_train, label_temp = train_test_split(
        X_features, attack_cat, label, test_size = 0.2, random_state = 1)
    X_test, X_val, attack_cat_test, attack_cat_val, label_test, label_val = train_test_split(
        X_temp, attack_cat_temp, label_temp, test_size = 0.5, random_state = 1)
    
    DTC = dtc(df=df, x_train=X_train, x_test=X_test, x_val=X_val, 
        label_train=label_train, label_test=label_test, label_val=label_val,
        attack_cat_train=attack_cat_train, attack_cat_test=attack_cat_test, attack_cat_val=attack_cat_val)
    
    DTC.get_experiment_data_part_one()
    DTC.optimal_training_selected_features_attack()
    DTC.optimal_training_selected_features_label()