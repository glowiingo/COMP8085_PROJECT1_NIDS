import sys

from sklearn import preprocessing
from Constants import *
from pandas import read_csv
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import knn
import pickle
import os.path
import pandas as pd
from sklearn import metrics
import DecisionTreeClassifier as dtc

## Variables
knn_name = 'K Nearest Neighbors'
dtc_name = 'Decision Tree'
lrc_name = 'Logistic Regression'

knn_arg = 'knn'
dtc_arg = 'dtc'
lrc_arg = 'lrc'

label_task = 'label'
attack_cat_task = 'attack_cat'

csv_ext = 'csv'
pkl_ext = 'pkl'

col_names = ['srcip', 'sport', 'dstip',	'dsport', 'proto', 'state',	'dur', 'sbytes', 'dbytes', 'sttl',	'dttl',	'sloss', 
        'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',	'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',	'ct_ftp_cmd', 'ct_srv_src',	'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']

def process_dataframe(df):
    df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)
    df.replace(r'\s+', '', regex=True, inplace=True)
    df.fillna('None', inplace=True)
    df.drop(0, inplace=True)
    feature_cols = list(df.columns[0:-2])

    categorical = df[feature_cols]
    enc = OrdinalEncoder()
    enc.fit(categorical)
    numerical = enc.transform(categorical)
    for n, feat in enumerate(feature_cols):
            df[feat] = numerical[:, n]
    return df

def print_prediction_report_for_label(X_test, y_test, classifier_label, normalize, convert_y):        
    X_test = X_test[SELECTED_FEATURES_LABEL_RFE]
    if normalize == True:
            scaler = StandardScaler()
            X_test = scaler.fit_transform(X_test)
    if convert_y == True:
            y_test = y_test.astype(int)
    y_label_pred = classifier_model.predict(X_test)
    print('Classifier: ' + classifier_label)
    print(metrics.classification_report(y_test, y_label_pred))

def print_prediction_report_for_attack_cat(X_test, y_test, classifier_label, normalize):
    X_test = X_test[SELECTED_FEATURES_ATTACK_CAT_RFE]    
    if normalize == True:
            scaler = StandardScaler()
            X_test= scaler.fit_transform(X_test)
    y_pred_attack_cat = classifier_model.predict(X_test)
    print('Classifier: ' + classifier_label)
    print(metrics.classification_report(y_test, y_pred_attack_cat))

print()

## CMD Arugments

## Check if there are enough arguments
if len(sys.argv) != 5:
    print('The program will terminate due to incorrect number of arguments.')    
    exit()
testing_file_name = sys.argv[1]
classifier = sys.argv[2].lower()
task = sys.argv[3].lower()
option_file_name = sys.argv[4]
option_file_list = option_file_name.split('.')
option_file_ext = option_file_list[len(option_file_list)-1]

## Check if arguments are valid
if os.path.isfile(testing_file_name) == False:
    print('The file ' + testing_file_name + ' does not exist.  The program will terminate')
    exit()

if os.path.isfile(option_file_name) == False:
    print('The file ' + option_file_name + ' does not exist.  The program will terminate')
    exit()

if classifier != knn_arg and classifier != dtc_arg and classifier != lrc_arg:
    print(classifier + ' is not a valid classifier.  The program will terminate')
    exit()
   
if task != label_task and task != attack_cat_task:
    print(task + ' is not a valid task.  The program will terminate')
    exit()
   
  

## Testing Data Set
testing_df = read_csv(testing_file_name, names=col_names, low_memory=False)
testing_df = process_dataframe(testing_df)
testing_X = testing_df[list(testing_df.columns[0:-2])]
testing_Y_label = testing_df['Label']
testing_Y_attack_cat = testing_df['attack_cat']

## Variable for holding the chosen classifier model
classifier_model = None

## Create model from CSV or PKL File
if option_file_ext != pkl_ext:
        ## Extract data set from training file and train models    
        training_df = read_csv(option_file_name, names = col_names, low_memory=False)
        training_df = process_dataframe(training_df)
        training_X = training_df[list(training_df.columns[0:-2])]
        
        training_y_label = training_df['Label']
        training_y_attack_cat = training_df['attack_cat']
        X_train, X_temp, y_label_train, y_label_temp, y_attack_cat_train, y_attack_cat_temp = train_test_split(training_X, training_y_label, training_y_attack_cat, test_size=0.2, random_state=42)        
        X_test, X_val, y_label_test, y_label_val, y_attack_cat_test, y_attack_cat_val = train_test_split(X_temp, y_label_temp, y_attack_cat_temp, test_size=0.5, random_state=42)
      
        if classifier == knn_arg:
                KNN = knn.knn()
                if task == label_task:
                        KNN.train_label_model(X_train, y_label_train)
                        classifier_model = KNN.get_label_model()            

                elif task == attack_cat_task:
                        KNN.train_attack_cat_model(X_train, y_attack_cat_train)
                        classifier_model = KNN.get_attack_cat_model()

        elif classifier == dtc_arg:
                DTC = dtc.dtc(x_train=X_train, x_test=X_test, x_val=X_val, 
                              label_train=y_label_train, label_test=y_label_test, label_val=y_label_val, 
                              attack_cat_train=y_attack_cat_train, attack_cat_test=y_attack_cat_test, attack_cat_val=y_attack_cat_val)
                if task == label_task:
                        classifier = DTC.train_model_selected_features_label_optimal()
                elif task == attack_cat_task:
                        classifier = DTC.train_model_selected_features_attack_optimal()
        elif classifier == lrc_arg:
                #LRC
                if task == 'label':
                        print('LRC.train_label_model(X_train, y_label_train)')
                        print('classifier_model = LRC.get_label_model')
                elif task == 'attack_cat':
                        print('LRC.train_label_model(X_train, y_attack_cat_train)')  
                        print('classifier_model = LRC.get_attack_cat_model')
else :
        ## Load models from pickle file    
        file = open(option_file_name, 'rb')
        knn_label = pickle.load(file)
        knn_attack_cat = pickle.load(file)
        dtc_label = pickle.load(file)
        dtc_attack_cat = pickle.load(file)
        lrc_label = pickle.load(file)
        lrc_attack_cat = pickle.load(file)
        file.close()       
        
        if classifier == knn_arg:
                KNN = knn.knn()
                if task == label_task:
                        classifier_model = knn_label
                elif task == attack_cat_task:
                        classifier_model = knn_attack_cat
        elif classifier == dtc_arg:
                # DTC
                if task == label_task:
                        classifier_model = dtc_label
                elif task == attack_cat_task:
                        classifier_model = dtc_attack_cat
        elif classifier == lrc_arg:
                #LRC
                if task == label_task:                        
                        classifier_model = lrc_label
                elif task == attack_cat_task:                        
                        classifier_model = lrc_attack_cat

## Print the report
if classifier == knn_arg:
        if task == label_task:
                print_prediction_report_for_label(testing_X, testing_Y_label, knn_name, True, False)
        elif task == attack_cat_task:
                print_prediction_report_for_attack_cat(testing_X, testing_Y_attack_cat, knn_name, True)
elif classifier == dtc_arg:        
        if task == label_task:
                print_prediction_report_for_label(testing_X, testing_Y_label, dtc_name, False, True)
        elif task == attack_cat_task:
                print_prediction_report_for_attack_cat(testing_X, testing_Y_attack_cat, dtc_name, False)
elif classifier == lrc_arg:
        if task == label_task:
                print_prediction_report_for_label(testing_X, testing_Y_label, lrc_name, True, True)
        elif task == attack_cat_task:
                print_prediction_report_for_attack_cat(testing_X, testing_Y_attack_cat, lrc_name, True)


