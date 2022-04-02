import data_extract
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV
import numpy as np
from time import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import GradientBoostingRegressor


class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 1================")
        return

    def model_1_run(self):
        model1='Ensemble--SGDRegression,lassoRegression,KNeighborsRegressor'
        print("Model 1:",model1)
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        estimators = [('SDG',SGDRegressor(max_iter=1000, tol=1e-3)),
                      ('lasso', LassoCV(cv=10, random_state=42,fit_intercept=False)),
                      ('knr', KNeighborsRegressor(n_neighbors=30, metric='euclidean'))]

        final_estimator = GradientBoostingRegressor(n_estimators=100, subsample=0.5, min_samples_leaf=50, max_features=1,
                                                    random_state=42)
        reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator)
        features= ['Dalc', 'Fedu', 'Medu', 'studytime','failures', 'address', 'sex', 'Fjob', 'Mjob', 'reason' , 'higher']
        target_variable = 'G3'
        formula = target_variable+'~'
        for h,i in enumerate(features):
            if i !=target_variable:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        # print(formula)
        formula += '**3'
        a = data_extract.data_extraction()
        y_train,X_train = a.extract(formula,True)
        begin  =  time()
        reg.fit(X_train, y_train.ravel())
        end = time()
        print(f'Use {end - begin:6f} s to train')
        y_pred = reg.predict(X_train)
        print('Ensemble : performance in training data:',mean_squared_error(y_train.ravel(), y_pred))


        # Evaluate learned model on testing data, and print the results.
        ## testing
        b = data_extract.data_extraction()
        y_test,X_test =b.extract(formula,False)
        pred = reg.predict(X_test)
        MSE = mean_squared_error(y_test.ravel(), pred)
        print("Mean squared error\t" + str(MSE))
        return


    def model_2_run(self):
        model2 ='ElasticNet'
        print("--------------------\nModel 2:",model2)
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        features= ['Dalc', 'Fedu', 'Medu', 'studytime','failures', 'address', 'Fjob', 'Mjob', 'reason','cohabitation','school','studytime','freetime','higher']
        target_variable = 'G3'
        formula = target_variable+'~'
        for h,i in enumerate(features):
            if i !=target_variable:
                if h==0:
                    formula +=i
                else:
                    formula += '+'+i
            else:
                continue
        # print(formula)
        formula += '**3'

        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        a = data_extract.data_extraction()
        y, x = a.extract(formula, True)
        regr = ElasticNetCV(cv=10, random_state=0, fit_intercept=False)
        begin = time()
        regr.fit(x, y.ravel())
        end = time()
        print(f'Elastic Net: Use {end - begin:6f} s to train')
        # print(regr.alpha_)
        # print(regr.l1_ratio_)
        pred1 = regr.predict(x)
        print('performance in training data:', mean_squared_error(y.ravel(), pred1))



        # Evaluate learned model on testing data, and print the results
        #
        a = data_extract.data_extraction()

        y_test, x_test = a.extract(formula, False)
        pred = regr.predict(x_test)
        MSE = mean_squared_error(y_test.ravel(), pred)

        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error\t" + str(MSE))
        return



if __name__ == "__main__":
    a = Task1()
    a.model_1_run()
