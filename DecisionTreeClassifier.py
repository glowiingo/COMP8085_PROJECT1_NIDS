from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from Constants import *
import time
import pickle
import logging

# Temp test running
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class dtc:
    def __init__(self, x_train, x_test, x_val, label_train, label_test, label_val, attack_cat_train, attack_cat_test, attack_cat_val):
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.label_train = label_train
        self.label_test = label_test
        self.label_val = label_val
        self.attack_cat_train = attack_cat_train
        self.attack_cat_test = attack_cat_test
        self.attack_cat_val = attack_cat_val
    
    """
    Save pickle function will not be utilized after running once with experiments to save the model
    However, as it may be useful in future iterations, will be left as a relic.
    """
    def save_pickle(self, model, filename=''):
        if filename == '':
            logging.exception("Error: File name is empty, please insert a filename!")
        else:
            with open(filename, 'wb') as file:
               pickle.dump(model, file)

    """
    Load pickle function will not be utilized after running once with experiments to check the model file has been properly saved and can be loaded.
    However, as it may be useful in future iterations, they will be left as a relic.
    """
    def load_pickle(self, filename=''):
        if filename == '':
            logging.exception("Error: Pickle file name is empty, please select a file to load!")
            pass
        else:
            with open(filename, 'rb') as file:
                loaded_model = pickle.load(file)
            return loaded_model

    # EXPERIMENT PART 1 - RUN CLASSIFIER WITHOUT SELECTED FEATURES
    def train_model_no_features_selected_label(self):
        clf = DecisionTreeClassifier()
        clf_label = clf.fit(self.x_train, self.label_train)
        return clf_label

    def no_selected_features_label(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score without using Selected Features for Label Prediction....")
        
        clf_label = self.train_model_no_features_selected_label()
        self.save_pickle(clf_label, 'dtc_no_selected_features_label_unoptimized.pkl')
        
        # clf_label = self.load_pickle('dtc_no_selected_features_label_unoptimized.pkl')
        
        label_pred = clf_label.predict(self.x_val)
        acc_score = metrics.accuracy_score(self.label_val, label_pred)*100
        print("Label Prediction Accuracy on Decision Tree Classifier with No Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.label_val, label_pred))
        end = time.perf_counter()
        print(f"Experiment on running DTC without using Selected Features for Label Prediction completed in {end - start:0.4f} seconds\n")

    def train_model_no_features_selected_attack_cat(self):
        clf = DecisionTreeClassifier()
        clf_attack = clf.fit(self.x_train, self.attack_cat_train)
        return clf_attack

    def no_selected_features_attack_cat(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score without using Selected Features for Attack Category Prediction....")
        
        clf_attack = self.train_model_no_features_selected_attack_cat()
        self.save_pickle(clf_attack, 'dtc_no_selected_features_attack_unoptimized.pkl')
        
        # clf_attack = self.load_pickle('dtc_no_selected_features_attack_unoptimized.pkl')
        
        attack_pred = clf_attack.predict(self.x_val)
        acc_score = metrics.accuracy_score(self.attack_cat_val, attack_pred)*100
        print("Attack Category Prediction Accuracy on Decision Tree Classifier with No Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.attack_cat_val, attack_pred, labels=ATTACK_CAT_STR_VALUES))
        end = time.perf_counter()
        print(f"Experiment on running DTC without using Selected Features for Attack Category Prediction completed in {end - start:0.4f} seconds\n")

    # EXPERIMENT PART 1 - RUN CLASSIFIER WTIH RFE SELECTED FEATURES
    def train_model_selected_features_label(self):
        clf = DecisionTreeClassifier()
        x_train_selected_label = self.x_train[SELECTED_FEATURES_LABEL_RFE]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        
        return clf_label

    def selected_features_label(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with Selected Features for Label...")

        clf_label = self.train_model_selected_features_label()
        self.save_pickle(clf_label, 'dtc_rfe_selected_features_label_unoptimized.pkl')
        
        # clf_label = self.load_pickle('dtc_rfe_selected_features_label_unoptimized.pkl')
        
        x_val_selected_label = self.x_val[SELECTED_FEATURES_LABEL_RFE]
        label_pred = clf_label.predict(x_val_selected_label)
        acc_score = metrics.accuracy_score(self.label_val, label_pred)*100
        print("Label Prediction Accuracy of RFE Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.label_val, label_pred))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features for Label Prediction completed in {end - start:0.4f} seconds\n")

    def train_model_selected_features_attack(self):
        clf = DecisionTreeClassifier()
        x_train_selected_attack = self.x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]
        clf_attack = clf.fit(x_train_selected_attack, self.attack_cat_train)
        
        return clf_attack

    def selected_features_attack(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with Selected Features for Attack Category...")
        x_val_selected_attack = self.x_val[SELECTED_FEATURES_ATTACK_CAT_RFE]
        
        clf_attack = self.train_model_selected_features_attack()
        self.save_pickle(clf_attack, 'dtc_rfe_selected_features_attack_unoptimized.pkl')
        
        # clf_attack = self.load_pickle('dtc_rfe_selected_features_attack_unoptimized.pkl')
        
        attack_pred = clf_attack.predict(x_val_selected_attack)
        acc_score = metrics.accuracy_score(self.attack_cat_val, attack_pred)*100
        print("Attack Category Prediction Accuracy of RFE Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.attack_cat_val, attack_pred, labels=ATTACK_CAT_STR_VALUES))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features for Attack Category Prediction completed in {end - start:0.4f} seconds\n")

    # EXPERIMENT PART 1 - RUN CLASSIFIER WTIH EBFI SELECTED FEATURES
    def train_model_selected_features_label_ebfi(self):
        clf = DecisionTreeClassifier()
        x_train_selected_label = self.x_train[SELECTED_FEATURES_LABEL_EBFI]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        return clf_label

    def selected_features_label_ebfi(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with EBFI Selected Features for Label...")
        
        clf_label = self.train_model_selected_features_label_ebfi()
        self.save_pickle(clf_label, 'dtc_ebfi_selected_features_label_unoptimized.pkl')
        
        # clf_label = self.load_pickle('dtc_ebfi_selected_features_label_unoptimized.pkl')
        
        x_val_selected_label = self.x_val[SELECTED_FEATURES_LABEL_EBFI]
        label_pred = clf_label.predict(x_val_selected_label)
        acc_score = metrics.accuracy_score(self.label_val, label_pred)*100
        print("Label Prediction Accuracy of EBFI Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.label_val, label_pred))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features from EBFI for Label Prediction completed in {end - start:0.4f} seconds\n")

    def train_model_selected_features_attack_ebfi(self):
        clf = DecisionTreeClassifier()
        x_train_selected_attack = self.x_train[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        clf_attack = clf.fit(x_train_selected_attack, self.attack_cat_train)
        return clf_attack

    def selected_features_attack_ebfi(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with EBFI Selected Features for Attack Category...")
        x_val_selected_attack = self.x_val[SELECTED_FEATURES_ATTACK_CAT_EBFI]
        
        clf_attack = self.train_model_selected_features_attack_ebfi()
        self.save_pickle(clf_attack, 'dtc_ebfi_selected_features_attack_unoptimized.pkl')
        
        # clf_attack = self.load_pickle('dtc_ebfi_selected_features_attack_unoptimized.pkl')
        
        attack_pred = clf_attack.predict(x_val_selected_attack)
        acc_score = metrics.accuracy_score(self.attack_cat_val, attack_pred)*100
        print("Attack Category Prediction Accuracy of EBFI Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.attack_cat_val, attack_pred, labels=ATTACK_CAT_STR_VALUES))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features for Attack Category Prediction completed in {end - start:0.4f} seconds\n")

    # EXPERIMENT PART 1 - RUN CLASSIFIER WTIH PCA SELECTED FEATURES
    def train_model_selected_features_label_pca(self):
        clf = DecisionTreeClassifier()
        x_train_selected_label = self.x_train[SELECTED_FEATURES_ATTACK_CAT_PCA]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        return clf_label

    def selected_features_label_pca(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with PCA Selected Features for Label...")
        
        clf_label = self.train_model_selected_features_label_pca()
        self.save_pickle(clf_label, 'dtc_pca_selected_features_label_unoptimized.pkl')
        
        # clf_label = self.load_pickle('dtc_pca_selected_features_label_unoptimized.pkl')
        
        x_val_selected_label = self.x_val[SELECTED_FEATURES_ATTACK_CAT_PCA]
        label_pred = clf_label.predict(x_val_selected_label)
        acc_score = metrics.accuracy_score(self.label_val, label_pred)*100
        print("Label Prediction Accuracy of PCA Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.label_val, label_pred))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features from PCA for Label Prediction completed in {end - start:0.4f} seconds\n")

    def train_model_selected_features_attack_pca(self):
        clf = DecisionTreeClassifier()
        x_train_selected_attack = self.x_train[SELECTED_FEATURES_ATTACK_CAT_PCA]
        clf_attack = clf.fit(x_train_selected_attack, self.attack_cat_train)
        
        return clf_attack

    def selected_features_attack_pca(self):
        start = time.perf_counter()
        print("Calculating Accuracy Score with EBFI Selected Features for Attack Category...")
        x_val_selected_attack = self.x_val[SELECTED_FEATURES_ATTACK_CAT_PCA]
        
        clf_attack = self.train_model_selected_features_attack_pca()
        self.save_pickle(clf_attack, 'dtc_pca_selected_features_attack_unoptimized.pkl')
        
        # clf_attack = self.load_pickle('dtc_pca_selected_features_attack_unoptimized.pkl')
        
        attack_pred = clf_attack.predict(x_val_selected_attack)
        acc_score = metrics.accuracy_score(self.attack_cat_val, attack_pred)*100
        print("Attack Category Prediction Accuracy of EBFI Selected Features: {:.2f}%\n".format(acc_score))
        print(metrics.classification_report(self.attack_cat_val, attack_pred, labels=ATTACK_CAT_STR_VALUES))
        end = time.perf_counter()
        print(f"Experiment on running DTC using Selected Features from PCA for Attack Category Prediction completed in {end - start:0.4f} seconds\n")

    # PART 2 - Optimized the training and classifier so that best possible scores are retrieved for Labels
    def train_model_selected_features_label_optimal(self):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=7)
        x_train_selected_label = self.x_train[SELECTED_FEATURES_LABEL_RFE]
        clf_label = clf.fit(x_train_selected_label, self.label_train)
        return clf_label
    
    def optimal_training_selected_features_label(self):
        start = time.perf_counter()
        x_test_selected_label = self.x_test[SELECTED_FEATURES_LABEL_RFE]
        
        clf_label = self.train_model_selected_features_label_optimal()
        self.save_pickle(clf_label, 'dtc_rfe_selected_features_label_optimized.pkl')
        
        # clf_label = self.load_pickle('dtc_rfe_selected_features_label_optimized.pkl')
        
        label_pred = clf_label.predict(x_test_selected_label)
        acc_score = metrics.accuracy_score(self.label_test, label_pred)*100
        print("Label Prediction Accuracy of RFE Selected Features Optimal: {:.2f}%\n".format(acc_score))
        classifier_name = "Decision Tree Classifier"
        print("\nClassifier: {}\n".format(classifier_name))
        print(metrics.classification_report(self.label_test, label_pred))
        end = time.perf_counter()
        print(f"Running DTC using Selected Features for Label Prediction with Hyper Parameter Adjustment completed in {end - start:0.4f} seconds\n")

    # PART 3 - Optimized the training and classifier so that best possible scores are retrieved for Attack Category
    def train_model_selected_features_attack_optimal(self):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=16)
        x_train_selected_attack_cat = self.x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]
        clf_attack_cat = clf.fit(x_train_selected_attack_cat, self.attack_cat_train)
        return clf_attack_cat
    
    def optimal_training_selected_features_attack(self):
        start = time.perf_counter()
        x_test_selected_attack_cat = self.x_test[SELECTED_FEATURES_ATTACK_CAT_RFE]
        
        clf_attack_cat = self.train_model_selected_features_attack_optimal()
        self.save_pickle(clf_attack_cat, 'dtc_rfe_selected_features_attack_optimized.pkl')
        
        # clf_attack_cat = self.load_pickle('dtc_rfe_selected_features_attack_optimized.pkl')
        
        attack_cat_pred = clf_attack_cat.predict(x_test_selected_attack_cat)
        acc_score = metrics.accuracy_score(self.attack_cat_test, attack_cat_pred)*100
        print("Attack Category Prediction Accuracy of RFE Selected Features Optimal: {:.2f}%\n".format(acc_score))
        classifier_name = "Decision Tree Classifier"
        print("\nClassifier: {}\n".format(classifier_name))
        print(metrics.classification_report(self.attack_cat_test, attack_cat_pred, labels=ATTACK_CAT_STR_VALUES))
        end = time.perf_counter()
        print(f"Running DTC using Selected Features for Attack Category Prediction with Hyper Parameter Adjustment completed in {end - start:0.4f} seconds\n")

    def get_experiment_data_part_one_dtc(self):
        print("============= PART ONE EXPERIMENTS =============")
        self.no_selected_features_label()
        self.no_selected_features_attack_cat()
        self.selected_features_label()
        self.selected_features_attack()
        print("============ EBFI SELECTED FEATURES EXPERIMENT ============")
        self.selected_features_label_ebfi()
        self.selected_features_attack_ebfi()
        print("============ PCA SELECTED FEATURES EXPERIMENT ============")
        self.selected_features_label_pca()
        self.selected_features_attack_pca()
        print("============= END OF PART ONE EXPERIMENTS =============\n")

def currPreprocess():
    col_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 
        'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']
    df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", names = col_names, low_memory=False, skipinitialspace=True)
    df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)
    df.replace(r'\s+', '', regex=True, inplace=True)
    df.fillna('None', inplace=True)
    df.drop(0, inplace=True)
    feature_cols = list(df.columns[0:-2])

    categorical = df[feature_cols]
    enc = preprocessing.OrdinalEncoder()
    enc.fit(categorical)
    numerical = enc.transform(categorical)
    for n, feat in enumerate(feature_cols):
        df[feat] = numerical[:, n]
    return df

def oldPreprocess():
    df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True)
    df = df.replace(r'\s+', '', regex=True)
    df.fillna('None', inplace=True)
    df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)

    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].astype('str')
    df['is_ftp_login'] = df['is_ftp_login'].astype('str')
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].astype('str')

    df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
    df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")

    # converting str to int
    # df['attack_cat'] = pd.factorize(df['attack_cat'])[0]
    df['proto'] = pd.factorize(df['proto'])[0]
    df['state'] = pd.factorize(df['state'])[0]
    df['service'] = pd.factorize(df['service'])[0]

    df['ct_flw_http_mthd'] = pd.factorize(df['ct_flw_http_mthd'])[0]
    df['is_ftp_login'] = pd.factorize(df['is_ftp_login'])[0]
    df['ct_ftp_cmd'] = pd.factorize(df['ct_ftp_cmd'])[0]

    df['srcip'] = preprocessing.LabelEncoder().fit_transform(df['srcip'])
    df['dstip'] = preprocessing.LabelEncoder().fit_transform(df['dstip'])
    return df

if __name__ == '__main__':
    df = currPreprocess()
    # df = oldPreprocess()

    feature_cols = list(df.columns[0:-2]) # remove label and attack_cat
    X_features = df[feature_cols] # Features
    label = df['Label'] # Variable
    attack_cat = df['attack_cat'] # Variable
    X_train, X_temp, attack_cat_train, attack_cat_temp, label_train, label_temp = train_test_split(
        X_features, attack_cat, label, test_size = 0.2, random_state = 1)
    X_test, X_val, attack_cat_test, attack_cat_val, label_test, label_val = train_test_split(
        X_temp, attack_cat_temp, label_temp, test_size = 0.5, random_state = 1)
    
    DTC = dtc(x_train=X_train, x_test=X_test, x_val=X_val, 
        label_train=label_train, label_test=label_test, label_val=label_val,
        attack_cat_train=attack_cat_train, attack_cat_test=attack_cat_test, attack_cat_val=attack_cat_val)
    
    DTC.get_experiment_data_part_one_dtc()
    DTC.optimal_training_selected_features_label()
    DTC.optimal_training_selected_features_attack()