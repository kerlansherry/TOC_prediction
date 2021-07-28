import math
import os
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    median_absolute_error
from tensorflow import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import scale, MinMaxScaler
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

data = pd.read_csv('data/TOC.csv',header=None)
data = data.values
row,column = data.shape
x0 = data[0:119,:]
x1 = data[119:238,:]
x2 = data[238:357,:]
print(x0.shape,x1.shape,x2.shape)
# print('x0:')
# print(x0)
# print('x1:')
# print(x1)

mm = MinMaxScaler((-1,1))
x0 = mm.fit_transform(x0)
x1 = mm.fit_transform(x1)
x2 = mm.fit_transform(x2)
clusters = (x0,x1,x2)

y_pred_array = np.zeros((1,1))
y_test_array = np.zeros((1,1))

# 不同的激活函数
# for activate_func in ['linear','sigmoid','tanh','relu']:
#     print('activate_func',activate_func)

# 不同的最后一层激活函数
# for final_func in ['linear', 'sigmoid', 'tanh', 'relu']:

# 不同的优化函数
# for optimal_func in ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'nadam']:

# 不同的学习率
for lr in [0.001,]:#,0.001,0.002,0.01,0.02,0.1,0.2,1]:
# default为 0.01
# lr = 0.01

    # 根据上面选择的情况，选择另外几个的缺省值（default）
    # activate_func default为 relu
    activate_func = 'relu'
    # final_func default为 tanh
    final_func = 'linear'
    # optimal_func default为 sgd
    # optimal_func = 'sgd'

    for cluster in clusters:

        X = cluster[:, 0:column - 1]
        Y = cluster[:, column - 1:column]
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        # 换不同的划分比例
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6)

        def build_model():
            # define model
            model = Sequential()
            model.add(Dense(10, input_dim=7, activation=activate_func))
            model.add(Dense(5, activation=activate_func))
            # model.add(Dense(64, activation=activate_func))
            # model.add(Dense(32, activation=activate_func))
            # model.add(Dense(16, activation=activate_func))
            # model.add(Dense(8, activation=activate_func))
            # model.add(Dense(4, activation=activate_func))
            # model.add(Dense(2, activation=activate_func))
            model.add(Dense(1, activation=final_func))
            rms = optimizers.RMSprop(lr=lr)
            model.compile(loss='mse', optimizer=rms, metrics=['mse'])
            # model.compile(loss='mse', optimizer=optimal_func, metrics=['mse'])
            # fit model
            # model.fit(trainX, trainy, epochs=50, verbose=0)
            return model

        def build_model2():
            # define model
            model = Sequential()
            # model.add(Dense(128, input_dim=7, activation=activate_func))
            model.add(Dense(10, input_dim=7, activation=activate_func))
            model.add(Dense(5, activation=activate_func))
            # model.add(Dense(64, activation=activate_func))
            # model.add(Dense(32, activation=activate_func))
            # model.add(Dense(16, activation=activate_func))
            # model.add(Dense(8, activation=activate_func))
            # model.add(Dense(4, activation=activate_func))
            # model.add(Dense(2, activation=activate_func))
            model.add(Dense(1, activation=final_func))
            rms = optimizers.RMSprop(lr=lr)
            model.compile(loss='mse', optimizer=rms, metrics=['mse'])
            # fit model
            # model.fit(trainX, trainy, epochs=50, verbose=0)
            return model

        def build_model3():
            # define model
            model = Sequential()
            # model.add(Dense(128, input_dim=7, activation=activate_func))
            model.add(Dense(10, input_dim=7, activation=activate_func))
            model.add(Dense(5, activation=activate_func))
            # model.add(Dense(64, activation=activate_func))
            # model.add(Dense(32, activation=activate_func))
            # model.add(Dense(16, activation=activate_func))
            # model.add(Dense(8, activation=activate_func))
            # model.add(Dense(4, activation=activate_func))
            # model.add(Dense(2, activation=activate_func))
            model.add(Dense(1, activation=final_func))
            rms = optimizers.RMSprop(lr=lr)
            model.compile(loss='mse', optimizer=rms, metrics=['mse'])
            # fit model
            # model.fit(trainX, trainy, epochs=50, verbose=0)
            return model

        def build_model4():
            model = Sequential()
            # model.add(Dense(128, input_dim=7, activation=activate_func))
            model.add(Dense(10, input_dim=7, activation=activate_func))
            model.add(Dense(5, activation=activate_func))
            # model.add(Dense(64, activation=activate_func))
            # model.add(Dense(32, activation=activate_func))
            # model.add(Dense(16, activation=activate_func))
            # model.add(Dense(8, activation=activate_func))
            # model.add(Dense(4, activation=activate_func))
            # model.add(Dense(2, activation=activate_func))
            model.add(Dense(1, activation=final_func))
            rms = optimizers.RMSprop(lr=lr)
            model.compile(loss='mse', optimizer=rms, metrics=['mse'])
            # fit model
            # model.fit(trainX, trainy, epochs=50, verbose=0)
            return model

        def build_model5():
            model = Sequential()
            # model.add(Dense(128, input_dim=7, activation=activate_func))
            model.add(Dense(10, input_dim=7, activation=activate_func))
            model.add(Dense(5, activation=activate_func))
            # model.add(Dense(64, activation=activate_func))
            # model.add(Dense(32, activation=activate_func))
            # model.add(Dense(16, activation=activate_func))
            # model.add(Dense(8, activation=activate_func))
            # model.add(Dense(4, activation=activate_func))
            # model.add(Dense(2, activation=activate_func))
            model.add(Dense(1, activation=final_func))
            rms = optimizers.RMSprop(lr=lr)
            model.compile(loss='mse', optimizer=rms, metrics=['mse'])
            # fit model
            # model.fit(trainX, trainy, epochs=50, verbose=0)
            return model

        epoch = 100
        batch_size = 10
        model = KerasRegressor(build_fn=build_model, epochs=epoch, batch_size=batch_size)
        model2 = KerasRegressor(build_fn=build_model2, epochs=2 * epoch, batch_size=batch_size)
        model3 = KerasRegressor(build_fn=build_model3, epochs=3 * epoch, batch_size=batch_size)
        model4 = KerasRegressor(build_fn=build_model4, epochs=4 * epoch, batch_size=batch_size)
        model5 = KerasRegressor(build_fn=build_model5, epochs=5 * epoch, batch_size=batch_size)
        import math
        from sklearn.metrics import mean_squared_error, r2_score

        # model.fit(X_train, y_train)
        # r21 = r2_score(y_test, model.predict(X_test))
        # R221 = 1 - math.sqrt(1 - r21)
        # Mse1 = mean_squared_error(y_test, model.predict(X_test))
        # Rmse1 = math.sqrt(Mse1)
        #
        # model2.fit(X_train, y_train)
        # r22 = r2_score(y_test, model2.predict(X_test))
        # R222 = 1 - math.sqrt(1 - r22)
        # Mse2 = mean_squared_error(y_test, model2.predict(X_test))
        # Rmse2 = math.sqrt(Mse2)
        #
        # model3.fit(X_train, y_train)
        # r23 = r2_score(y_test, model3.predict(X_test))
        # R223 = 1 - math.sqrt(1 - r23)
        # Mse3 = mean_squared_error(y_test, model3.predict(X_test))
        # Rmse3 = math.sqrt(Mse3)
        #
        # model4.fit(X_train, y_train)
        # r24 = r2_score(y_test, model4.predict(X_test))
        # R224 = 1 - math.sqrt(1 - r24)
        # Mse4 = mean_squared_error(y_test, model4.predict(X_test))
        # Rmse4 = math.sqrt(Mse4)
        #
        # model5.fit(X_train, y_train)
        # r25 = r2_score(y_test, model5.predict(X_test))
        # R225 = 1 - math.sqrt(1 - r25)
        # Mse5 = mean_squared_error(y_test, model.predict(X_test))
        # Rmse5 = math.sqrt(Mse5)

        regs2 = [model, model2, model3, model4, model5]

        X_train_stack = np.zeros((X_train.shape[0], len(regs2)))
        X_test_stack = np.zeros((X_test.shape[0], len(regs2)))
        ### 5折stacking
        n_folds = 5
        skf = KFold(n_splits=n_folds)
        for i, reg in enumerate(regs2):
            #     print("分类器：{}".format(clf))
            X_stack_test_n = np.zeros((X_test.shape[0], n_folds))
            for j, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
                tr_x = X_train[train_index]
                tr_y = y_train[train_index]
                reg.fit(tr_x, tr_y)
                # 生成stacking训练数据集
                X_train_stack[test_index, i] = reg.predict(X_train[test_index])
                print(X_train_stack.shape)
                X_stack_test_n[:, j] = reg.predict(X_test)
                print(X_stack_test_n.shape)
            # 生成stacking测试数据集
            X_test_stack[:, i] = X_stack_test_n.mean(axis=1)
            print(X_test_stack)
        ###第二层模型LR
        from sklearn.linear_model import LogisticRegression, LinearRegression

        clf_second = LinearRegression()
        clf_second.fit(X_train_stack, y_train)
        pred = clf_second.predict(X_test_stack)
        print('pred:', pred.shape)
        r2 = r2_score(y_test, pred)
        from sklearn.metrics import mean_squared_error
        import math

        R22 = 1 - math.sqrt(1 - r2)
        Mse = mean_squared_error(y_test, pred)
        Rmse = math.sqrt(Mse)
        # np.savetxt('result/y_test8.csv', y_test, delimiter=',')
        # np.savetxt('result/y_pred8.csv', pred, delimiter=',')
        # print(r21, R221, Mse1, Rmse1)
        # print(r22, R222, Mse2, Rmse2)
        # print(r23, R223, Mse3, Rmse3)
        # print(r24, R224, Mse4, Rmse4)
        # print(r25, R225, Mse5, Rmse5)
        print(r2, R22, Mse, Rmse)
        print(y_pred_array.shape, pred.shape)


        y_pred_array = np.vstack((y_pred_array, pred))
        y_test_array = np.vstack((y_test_array, y_test))
    np.savetxt('result/y_pred.csv', y_pred_array, delimiter=',')
    np.savetxt('result/y_test.csv', y_test_array, delimiter=',')

    y_pred_array = np.delete(y_pred_array, 0, 0)
    y_test_array = np.delete(y_test_array, 0, 0)

    # 计算评价指标
    R2 = r2_score(y_test_array, y_pred_array, multioutput='raw_values')             #拟合优度
    R22 = 1-math.sqrt(1-R2)
    Mse = mean_squared_error(y_test_array,y_pred_array)                                 #均方差
    Mae = mean_absolute_error(y_test_array,y_pred_array,
                            sample_weight=None,
                            multioutput='uniform_average')                              #平均绝对误差
    Variance =  explained_variance_score(y_test_array, y_pred_array,
                                  sample_weight=None,
                                   multioutput='uniform_average')                        #可释方差得分
    Meae =  median_absolute_error(y_test_array, y_pred_array)                       #中值绝对误差
    Smape = smape(y_test_array, y_pred_array)
    print ("R2 :%.4f" %  R2)
    print ("R22 :%.4f" %  R22)
    print ("Mse :%.4f" %  Mse)
    print ("Rmse :%.4f" %  math.sqrt(Mse))
    print ("Mae :%.4f" %  Mae)
    print("Smape :%.4f" % Smape)
    s = [R2, Mse, Mae, Smape]
    # print ("Variance :%.4f" %  Variance)
    # print ("Meae :%.4f" %  Meae)
    if os.path.exists('result') is False:
        os.makedirs('result')

    # 根据40行及以下的情况选择对应的
    # np.savetxt('result/activate_func-'+activate_func+'.csv', s, delimiter = ',')
    # np.savetxt('result/final_func-' + final_func + '.csv', s, delimiter=',')
    # np.savetxt('result/optimal_func-' + optimal_func + '.csv', s, delimiter=',')
    # np.savetxt('result/lr-' + str(lr) + '.csv', s, delimiter=',')