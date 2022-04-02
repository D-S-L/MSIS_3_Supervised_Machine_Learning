
from sklearn.neural_network import MLPClassifier
import data_extract2 as data_extract
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from time import time
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        return

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        model_name = 'Neural Network'
        print("Model 1:", model_name)
        # Train the model 1 with your best hyper parameters (if have) and features on training data.

        clf = MLPClassifier(solver='sgd', alpha=10,
                            hidden_layer_sizes=(13, 10,10), random_state=1, max_iter=2000, warm_start=True)

        features=['school', 'sex', 'Mjob','age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel','freetime', 'goout','Dalc', 'Walc', 'health', 'absences', 'G3']
        features =['address', 'sex', 'Fjob', 'Mjob', 'reason','higher','Medu', 'Fedu','Dalc','studytime','failures']
        target_variable = 'Mjob'
        formula = target_variable+'~('
        for h,i in enumerate(features):
            if i !=target_variable:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        # print(formula)
        formula += ')**3'
        a = data_extract.data_extraction()
        y_train, X_train = a.extract(formula, True)

        begin = time()
        for i in range(15):
            clf.fit(X_train, y_train)
        end = time()
        print(f'Use {end - begin:6f} s to train')
        # y_pred = clf.predict(X_train)

        # overall_acc_train = accuracy_score(y_train, y_pred)
        # print(overall_acc_train)

        # Evaluate learned model on testing data, and print the results.

        a = data_extract.data_extraction()
        y_test, x_test,mjob_position = a.extract(formula, False)
        pred = clf.predict(x_test)


        overall_acc_test = accuracy_score(y_test, pred)
        precision,recall,fscore,support=precision_recall_fscore_support(y_test, pred, average='macro',zero_division=0)


        self.print_macro_results(overall_acc_test,fscore, precision,recall)
        categories = ["teacher", "health", "service", "at_home", "other"]
        for category in categories:
            cur_predict = [1 if i==category else 0 for i in pred]
            cur_true = [1 if i==category else 0 for i in y_test]
            cur_precision, cur_recall, cur_fscore, _ = precision_recall_fscore_support(cur_predict, cur_true,average ='binary',zero_division=0)
            self.print_category_results(category, cur_fscore,cur_precision, cur_recall)
        return



    def model_2_run(self):
        model2_name = 'SVM--OneVsRest'
        print("--------------------\nModel 2:", model2_name)
        # Train the model 2 with your best hyper parameters (if have) and features on training data.

        features=['school', 'sex','Mjob', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel','freetime', 'goout','Dalc', 'Walc', 'health', 'absences', 'G3']
        features=['address', 'sex', 'Fjob', 'Mjob', 'reason','higher','Medu', 'Fedu','Dalc','studytime','failures']
        target_variable = 'Mjob'
        formula = target_variable+'~('
        for h,i in enumerate(features):
            if i !=target_variable:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        # print(formula)
        formula += ')**3'
        a = data_extract.data_extraction()
        y_train, X_train = a.extract(formula, True)

        begin = time()

        # clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=200)

        # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        # clf.fit(X_train, y_train)
        clf = OneVsRestClassifier(SVC(kernel='sigmoid')).fit(X_train, y_train)

        end = time()
        print(f'Use {end - begin:6f} s to train')
        y_pred = clf.predict(X_train)
        # overall_acc_train = accuracy_score(y_train, y_pred)
        # print(overall_acc_train)



        # Evaluate learned model on testing data, and print the results.
        a = data_extract.data_extraction()
        y_test, x_test,_ = a.extract(formula, False)
        pred = clf.predict(x_test)

        overall_acc_test = accuracy_score(y_test, pred)
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, pred, average='macro',zero_division=0)


        self.print_macro_results(overall_acc_test, fscore,precision, recall)
        categories = ["teacher", "health", "service", "at_home", "other"]

        for category in categories:
            cur_predict = [1 if i==category else 0 for i in pred]
            cur_true = [1 if i==category else 0 for i in y_test]
            cur_precision, cur_recall, cur_fscore, _ = precision_recall_fscore_support(cur_predict, cur_true,average ='binary',zero_division=0)
            self.print_category_results(category, cur_fscore,cur_precision, cur_recall)
        return
