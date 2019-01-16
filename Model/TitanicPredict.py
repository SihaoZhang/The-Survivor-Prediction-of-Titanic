# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:24:33 2019

@author: Sihao
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# New function to return name title only
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'

# New function to classify each title into 1 of 4 overarching titles
def set_title(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady','Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def z_score(x, axis):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x

def dataToCsv(file, data, columns, target):
    data = list(data)
    columns = list(columns)
    file_data = pd.DataFrame(data, index=range(len(data)), columns=columns)
    file_target = pd.DataFrame(target, index=range(len(data)), columns=['target'])
    file_all = file_data.join(file_target, how='outer')
    file_all.to_csv(file)

def preprocess(dataset):
    # Drop useless features
    dataset.drop(labels=["PassengerId", "Cabin", "Ticket"], axis=1, inplace=True)

    # Replace NaN
    dataset["Embarked"].fillna("S", inplace=True)
    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)
    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)

    # Numerlizartion
    le_sex = LabelEncoder()
    le_sex.fit(dataset["Sex"])
    dataset["Sex"] = le_sex.transform(dataset["Sex"])
    le_embarked = LabelEncoder()
    le_embarked.fit(dataset["Embarked"])
    dataset["Embarked"] = le_embarked.transform(dataset["Embarked"])

    # Turning data into 4 bins due to heavy skew in data
    # dataset['Age'] = pd.qcut(dataset['Age'], 4, duplicates='drop')
    # Transforming bins to numbers
    # lbl = LabelEncoder()
    # dataset['Age'] = lbl.fit_transform(dataset['Age'])

    # Applying the get_title function to create the new 'Title' feature
    # dataset['Title'] = dataset['Name'].map(lambda x: get_title(x))
    # dataset['Title'] = dataset.apply(set_title, axis=1)
    # dataset['Title'] = dataset['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master'], [0, 1, 2, 3])

    # Create new features
    dataset["Family"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset["Alone"] = dataset.Family.apply(lambda x: 1 if x == 1 else 0)

    # Scale
    scaler = StandardScaler()
    ages_train = np.array(dataset["Age"]).reshape(-1, 1)
    fares_train = np.array(dataset["Fare"]).reshape(-1, 1)
    dataset["Age"] = scaler.fit_transform(ages_train)
    dataset["Fare"] = scaler.fit_transform(fares_train)

    # dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    dataset.drop(labels=["Name"], axis=1, inplace=True)
    return dataset

def feature_selection(dataset, label):
    m, n = np.shape(dataset)
    prr = []
    for i in range(n):
        prr.append(list(pearsonr(label, dataset[:, i])))
    prr = np.array(prr)[:, 0]
    return prr

def accuracy(prediction):
    result = pd.read_csv(r"..\Data\gender_submission.csv")
    result = result['Survived'].values
    error = 0
    for i in range(len(result)):
        if result[i] != prediction[i]:
            error += 1
    pred_acc = error / len(result)
    return pred_acc


if __name__ == "__main__":
    data = pd.read_csv(r"..\Data\train.csv")
    data = preprocess(data)
    data_test = pd.read_csv(r"..\Data\test.csv")
    data_test = preprocess(data_test)

    #feature selection
    features = data.drop(labels=["Survived"], axis=1).values
    label = data["Survived"].values

    prr = feature_selection(features, label)
    print(prr)
    # f_features = np.delete(features, np.where(prr <= 0.6), axis=1)
    f_features = features
    # f_label = f_features[:, 1]
    f_label = label
    # f_features = preprocessing.scale(f_features[:, 1:])
    # test_features = np.delete(data_test.values, np.where(prr <= 0.6), axis=1)
    test_features = data_test.values

    from sklearn.model_selection import train_test_split  # to create validation data set

    X_training, X_valid, y_training, y_valid = train_test_split(f_features, f_label, test_size=0.20,
                                                                random_state=0)  # X_valid and y_valid are the validation sets

    # Random Forests
    random_forest = RandomForestClassifier(n_estimators=180, criterion='gini', max_features='sqrt',
                                             max_depth=3, min_samples_split=4, min_samples_leaf=2,
                                             n_jobs=50, random_state=29, verbose=1)
    random_forest.fit(X_training, y_training)
    Y_pred = random_forest.predict(X_valid)
    acc_svc = accuracy_score(y_valid, Y_pred)
    print("RandomForestClassifier_trainAcc:", acc_svc)
    ##############################################################################################################
    #SVM classifier
    svc_clf = svm.SVC()
    parameters_svc = {"kernel": ["rbf", "linear", "poly"], "probability": [True, False, False], "verbose": [True, False, False]}

    grid_svc = GridSearchCV(svc_clf, parameters_svc, scoring=make_scorer(accuracy_score))
    grid_svc.fit(X_training, y_training)

    svc_clf = grid_svc.best_estimator_

    svc_clf.fit(X_training, y_training)
    pred_svc = svc_clf.predict(X_valid)
    acc_svc = accuracy_score(y_valid, pred_svc)
    print("SVM_trainAcc:", acc_svc)
    ##############################################################################################################
    linsvc_clf = svm.LinearSVC()

    parameters_linsvc = {"multi_class": ["ovr", "crammer_singer"], "fit_intercept": [True, False],
                         "max_iter": [100, 500, 1000, 1500]}

    grid_linsvc = GridSearchCV(linsvc_clf, parameters_linsvc, scoring=make_scorer(accuracy_score))
    grid_linsvc.fit(X_training, y_training)

    linsvc_clf = grid_linsvc.best_estimator_

    linsvc_clf.fit(X_training, y_training)
    pred_linsvc = linsvc_clf.predict(X_valid)
    acc_linsvc = accuracy_score(y_valid, pred_linsvc)

    print("The Score for LinearSVC is: " + str(acc_linsvc))
    ##############################################################################################################
    dt_clf = DecisionTreeClassifier()

    parameters_dt = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"],
                     "max_features": ["auto", "sqrt", "log2"]}

    grid_dt = GridSearchCV(dt_clf, parameters_dt, scoring=make_scorer(accuracy_score))
    grid_dt.fit(X_training, y_training)

    dt_clf = grid_dt.best_estimator_

    dt_clf.fit(X_training, y_training)
    pred_dt = dt_clf.predict(X_valid)
    acc_dt = accuracy_score(y_valid, pred_dt)

    print("The Score for Decision Tree is: " + str(acc_dt))
    ##############################################################################################################
    gnb_clf = GaussianNB()

    parameters_gnb = {}

    grid_gnb = GridSearchCV(gnb_clf, parameters_gnb, scoring=make_scorer(accuracy_score))
    grid_gnb.fit(X_training, y_training)

    gnb_clf = grid_gnb.best_estimator_

    gnb_clf.fit(X_training, y_training)
    pred_gnb = gnb_clf.predict(X_valid)
    acc_gnb = accuracy_score(y_valid, pred_gnb)

    print("The Score for Gaussian NB is: " + str(acc_gnb))
    ##############################################################################################################
    knn_clf = KNeighborsClassifier()

    parameters_knn = {"n_neighbors": [3, 5, 10, 15], "weights": ["uniform", "distance"],
                      "algorithm": ["auto", "ball_tree", "kd_tree"],
                      "leaf_size": [20, 30, 50]}

    grid_knn = GridSearchCV(knn_clf, parameters_knn, scoring=make_scorer(accuracy_score))
    grid_knn.fit(X_training, y_training)

    knn_clf = grid_knn.best_estimator_

    knn_clf.fit(X_training, y_training)
    pred_knn = knn_clf.predict(X_valid)
    acc_knn = accuracy_score(y_valid, pred_knn)

    print("The Score for KNeighbors is: " + str(acc_knn))
    ##############################################################################################################
    from xgboost import XGBClassifier

    gbm = XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1).fit(X_training,
                                                                                                               y_training)
    predictions = gbm.predict(X_valid)
    pred_gbm = gbm.predict(X_valid)
    acc_gbm = accuracy_score(y_valid, pred_gbm)
    print("The Score for XGBClassifier is: " + str(acc_gbm))
    ##############################################################################################################
    clf = VotingClassifier(estimators = [('rf', random_forest),
                                         ('sc', svc_clf),
                                         ('dc', dt_clf),
                                         ('gc', gnb_clf),
                                         ('kc', knn_clf),
                                         ('xgb', gbm)], voting = 'soft', n_jobs = -1)

    clf.fit(f_features, f_label)
    prediction = clf.predict(test_features)
    print(type(prediction))
    prediction = pd.DataFrame(prediction, columns=['Survived'])
    prediction.to_csv("Titanic_predict.csv", index=False)




