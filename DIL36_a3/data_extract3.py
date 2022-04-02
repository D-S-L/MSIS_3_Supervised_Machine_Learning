import pandas as pd
import numpy as np
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler


class data_extraction():
    def __init__(self):
        self.sub_path = 'assign3_students_'
        return
    def construct_label(self,y_data):
        set_label = set()
        for i in y_data:
            for j in i.split(' '):
                set_label.add(j)

        label_positin = {}
        for i, j in enumerate(set_label):
            label_positin[j]=i

        y = [[0]*len(set_label) for _ in range(len(y_data))]
        for h,i in enumerate(y_data):
            for j in i.split(' '):
                y[h][label_positin[j]]=1
        return y


    def extract(self, formula, train):
        ### return x and y for training  or testing


        if train == True:
            # extract data for training
            path = self.sub_path+'train.txt'
            data = pd.read_csv(path, header=None, sep='\t')
            data.columns = ['school', 'sex', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu','Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel','freetime', 'goout','Dalc', 'Walc', 'health', 'absences', 'G3']
            label = formula.split('~')[0]

            data_label = list(data[label])
            y = self.construct_label(data_label)


            ## standardization
            data_num = data.select_dtypes('number').copy()
            tem_col = data.select_dtypes('number').columns
            data_num = pd.DataFrame(StandardScaler().fit_transform(data_num))
            data_num.columns = tem_col

            data_c = data.select_dtypes('object')

            data = pd.concat([data_num, data_c], axis=1)

            # self.skew =[self.data.select_dtypes('number').skew())]
            _, X = dmatrices(formula + ' - 1', data=data)
            return y, X



        else:
            # extract data for testing
            path_train = self.sub_path + 'train.txt'
            path_test = self.sub_path + 'test.txt'

            train = pd.read_csv(path_train, header=None, sep='\t')
            train.columns = ['school', 'sex', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Mjob','Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout','Dalc', 'Walc', 'health', 'absences', 'G3']



            test = pd.read_csv(path_test, header=None, sep='\t')
            test.columns = ['school', 'sex', 'age', 'address', 'familySize', 'cohabitation', 'Medu', 'Fedu', 'Mjob','Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport','nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc','health', 'absences', 'G3']
            label = formula.split('~')[0]

            data_label = list(test[label])
            y = self.construct_label(data_label)


            test_num =test.select_dtypes('number').copy()
            tem_col = test.select_dtypes('number').columns
            test_num = pd.DataFrame(StandardScaler().fit_transform(test_num))
            test_num.columns = tem_col

            test_c = test.select_dtypes('object')

            test= pd.concat([test_num, test_c], axis=1)


            total = [test, train]
            total = pd.concat(total)

            _, X = dmatrices(formula + ' - 1', data=total)
            X = X[:test.shape[0], :]

            return y, X








