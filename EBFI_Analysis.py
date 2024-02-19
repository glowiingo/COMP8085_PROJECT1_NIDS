import numpy as np
import matplotlib.pyplot as plt

# Temp test running
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

LABEL_INFOGAIN_PNG = 'label_infogain.png'
ATTACK_INFOGAIN_PNG = 'attack_cat_infogain.png'

class ebfi_analysis:
    def __init__(self, x_train, label_train, attack_cat_train):
        """
        Initializes the data required for Entropy Based Feature Importance Analysis
        
        x_train: training data including all columns but attack_cat and label
        label_train: only label column training data
        attack_cat train: only attack_cat column training data
        """
        self.x_train = x_train
        self.label_train = label_train
        self.attack_cat_train = attack_cat_train

        # function calls
        label_infogain_dict = self.get_label_infogain()
        attack_cat_infogain_dict = self.get_attack_cat_infogain()
        self.plot(label_infogain_dict=label_infogain_dict, attack_cat_infogain_dict=attack_cat_infogain_dict)
    
    def entropy(self, y):
        """
        Calculates the sum of probabilities of 
        each class inside of specified column
        y: the column that is targeted for calculation

        Code from: https://winder.ai/entropy-based-feature-selection/
        """
        probs = [] # probabilities of each class in the column
        for value in set(y):
            num_same_value = sum(y == value)
            p = num_same_value / len(y) # probability of specified value
            probs.append(p)

        return sum(-p * np.log2(p) for p in probs)

    def class_probability(self, feature, y):
        """
        Helper function that calculates the 
        proportional length of each value in the set of instances

        Code from: https://winder.ai/entropy-based-feature-selection/
        """
        probs = []
        for value in set(feature):
            select = feature == value # Split by feature value into two classes
            y_new = y[select]         # Those that exist in this class are now in y_new
            probs.append(float(len(y_new))/len(self.x_train))  # Convert to float, because ints don't divide well
        return probs

    def class_entropy(self, feature, y):
        """
        Helper function that calculates the entropy for 
        each value in the set of instances

        Code from: https://winder.ai/entropy-based-feature-selection/
        """
        ents = []
        for value in set(feature):
            select = feature == value # Split by feature value into two classes
            y_new = y[select]         # Those that exist in this class are now in y_new
            ents.append(self.entropy(y_new))
        return ents

    def proportionate_class_entropy(self, feature, y):
        """
        Helper function that calculatates the weighted proportional entropy 
        for a feature when splitting on all values

        Code from: https://winder.ai/entropy-based-feature-selection/
        """
        probs = self.class_probability(feature, y)
        ents = self.class_entropy(feature, y)
        return sum(np.multiply(probs, ents)) # Information gain equation

    def get_label_entropy(self):
        return self.entropy(self.label_train)
    
    def get_attack_cat_entropy(self):
        return self.entropy(self.attack_cat_train)

    def get_label_infogain(self):
        print("Running code for getting information gain data on all feature columns for EBFI Analysis on Labels...")
        label_infogain_dict = {}
        label_entropy = self.get_label_entropy()
        print("LABEL ENTROPY: ", label_entropy)
        for c in self.x_train.columns:
            new_entropy = self.proportionate_class_entropy(self.x_train[c], self.label_train)
            label_infogain_dict[c] = label_entropy - new_entropy
            print("%s %.5f" % (c, label_infogain_dict[c]))
        return label_infogain_dict
    
    def get_attack_cat_infogain(self):
        print("Running code for getting information gain data on all feature columns for EBFI Analysis on Attack Category...")
        attack_cat_infogain_dict = {}
        attack_cat_entropy = self.get_attack_cat_entropy()
        print("ATTACK_CAT ENTROPY: ", attack_cat_entropy)
        for c in self.x_train.columns:
            new_entropy = self.proportionate_class_entropy(self.x_train[c], self.attack_cat_train)
            attack_cat_infogain_dict[c] = attack_cat_entropy - new_entropy
            print("%s %.5f" % (c, attack_cat_infogain_dict[c]))
        return attack_cat_infogain_dict

    def plot(self, label_infogain_dict, attack_cat_infogain_dict):
        """
        Plotting label amd attack category information gain bar graph on all features
        """
        label_sorted_indices = np.argsort(list(label_infogain_dict.values()))[::-1]
        sorted_label_names = [list(label_infogain_dict.keys())[i] for i in label_sorted_indices]
        sorted_label_infogain = [list(label_infogain_dict.values())[i] for i in label_sorted_indices]

        fig = plt.figure(figsize = (10, 15))
        plt.bar(sorted_label_names, sorted_label_infogain, color ='steelblue', width = 0.5)
        plt.xticks(rotation='vertical')
        plt.title('Entropy Based Feature Analysis for Label')
        plt.xlabel('Features')
        plt.ylabel('Information Gain for Label')
        plt.savefig(LABEL_INFOGAIN_PNG, bbox_inches="tight")
        plt.show()
        print("Saved label information gain bar plot in: {}".format(LABEL_INFOGAIN_PNG))

        attack_cat_sorted_indices = np.argsort(list(attack_cat_infogain_dict.values()))[::-1]
        sorted_attack_cat_names = [list(attack_cat_infogain_dict.keys())[i] for i in attack_cat_sorted_indices]
        sorted_attack_cat_infogain = [list(attack_cat_infogain_dict.values())[i] for i in attack_cat_sorted_indices]

        fig = plt.figure(figsize = (10, 15))
        plt.bar(sorted_attack_cat_names, sorted_attack_cat_infogain, color ='rosybrown', width = 0.5)
        plt.xticks(rotation='vertical')
        plt.title('Entropy Based Feature Analysis for Attack Category')
        plt.xlabel('Features')
        plt.ylabel('Information Gain for Attack Category')
        plt.savefig(ATTACK_INFOGAIN_PNG, bbox_inches="tight")
        plt.show()
        print("Saved attack infomation gain bar plot in: {}".format(ATTACK_INFOGAIN_PNG))


if __name__ == '__main__':
    df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True)
    df = df.replace(r'\s+', '', regex=True)
    df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)
    attack_cat_str_values = df['attack_cat'].unique()

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
    
    ebfi_analysis(x_train=X_train, label_train=label_train, attack_cat_train=attack_cat_train)