from sklearn.neural_network import MLPClassifier
import data_extract3 as data_extract
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from time import time
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV

class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        return

    def model_1_run(self):
        model_1='Elastic net--OneVsRest'
        print("Model 1:",model_1)
        # Train the model 1 with your best hyper parameters (if have) and features on training data.

        features = ['school', 'sex', 'Mjob', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Fjob',
                    'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher',
                    'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3']
        features = ['address', 'sex', 'Fjob', 'Mjob', 'reason', 'higher', 'Medu', 'Fedu', 'Dalc', 'studytime',
                  'failures']
        target = 'edusupport'
        formula = target+'~('
        for h,i in enumerate(features):
            if i !=target:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        formula +=')**1'
        # print(formula)
        a = data_extract.data_extraction()
        y_train, X_train = a.extract(formula, True)

        begin = time()

        # clf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=200)

        # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        # clf.fit(X_train, y_train)
        # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=2, random_state=0)
        # clf = AdaBoostClassifier(n_estimators=100)
        # clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
        l1_ratios = np.linspace(0,1,10)
        clf = LogisticRegressionCV(cv=10, random_state=0,penalty='elasticnet',fit_intercept =False,solver='saga',l1_ratios=l1_ratios,max_iter=2000)

        clf = OneVsRestClassifier(clf).fit(X_train, y_train)

        end = time()
        print(f'Use {end - begin:6f} s to train')
        y_pred = clf.predict(X_train)
        # overall_acc_train = accuracy_score(y_train, y_pred)
        # print(overall_acc_train)

        # Evaluate learned model on testing data, and print the results.

        a = data_extract.data_extraction()
        y_test, x_test = a.extract(formula, False)
        pred = clf.predict(x_test)

        overall_acc_test = accuracy_score(np.array(y_test),(pred))

        ham = hamming_loss(np.array(y_train), (y_pred))


        print("Accuracy\t" + str(overall_acc_test) + "\tHamming loss\t" + str(ham))
        return

    def model_2_run(self):
        model_2 = 'SVM--OneVsRest'
        print("--------------------\nModel 2:",model_2)
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=500)

        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        features=['school', 'sex','Mjob', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel','freetime', 'goout','Dalc', 'Walc', 'health', 'absences', 'G3']
        features=['address', 'sex', 'Fjob', 'Mjob', 'reason','higher','Medu', 'Fedu','Dalc','studytime','failures']
        target = 'edusupport'
        formula = target+'~('
        for h,i in enumerate(features):
            if i !=target:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        formula +=')**2'
        # print(formula)
        a = data_extract.data_extraction()
        y_train, X_train = a.extract(formula, True)

        begin = time()

        # clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=200)

        # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        # clf.fit(X_train, y_train)
        # clf = OneVsRestClassifier(clf).fit(X_train, y_train)
        clf = OneVsRestClassifier(SVC(kernel='rbf')).fit(X_train, y_train)
        end = time()
        print(f'Use {end - begin:6f} s to train')
        y_pred = clf.predict(X_train)
        # overall_acc_train = accuracy_score(y_train, y_pred)
        # print(overall_acc_train)

        # Evaluate learned model on testing data, and print the results.

        a = data_extract.data_extraction()
        y_test, x_test = a.extract(formula, False)
        pred = clf.predict(x_test)

        overall_acc_test = accuracy_score(y_test, pred)
        ham = hamming_loss(y_train, y_pred)


        print("Accuracy\t" + str(overall_acc_test) + "\tHamming loss\t" + str(ham))
        return

