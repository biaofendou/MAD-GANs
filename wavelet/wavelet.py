import numpy as np
import pywt
import matplotlib.pyplot as plt
import swat
import time
import random
from collections import Counter

start = time.time()
from numpy import *
from sklearn.preprocessing import LabelEncoder
from sklearn import svm, preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# X, Y,data= swat.swat()


def pprint(msg):
    print(msg)
    # sys.stderr.write(msg+'\n')


# 按照划窗的大小将数据集进行处理，标签为一组数据中出现次数最多的数字
def slidingFunc(window_size, data, label):
    newdata = []
    newlabel = []
    L = len(data)
    interval = 1
    index = 0
    newdata_count = 0
    initial_value = -999
    while index + window_size < L:
        newdata.append(initial_value)
        newlabel.append(initial_value)
        sequence = []
        sequence_label = []
        for i in range(window_size):
            sequence.append(data[index + i])
            sequence_label.append(label[index + i])
            # newlabel[newdata_count] = label[index+i]
        newlabel[newdata_count] = Counter(sequence_label).most_common(1)[0][0]  # 出现次数最多的数
        index += interval
        newdata[newdata_count] = sequence
        newdata_count += 1
    return np.array(newdata), np.array(newlabel)


# 统计正样本结果的索引
def returnPositiveIndex(data, negative_sign):
    temp = []
    for i in range(len(data)):
        try:
            if int(data[i]) != negative_sign:
                temp.append(i)
        except:
            if int(data[i, -1]) != negative_sign:
                temp.append(i)
    return np.array(temp)


# 统计负样本结果的索引
def returnNegativeIndex(data, negative_sign):
    temp = []
    for i in range(len(data)):
        try:
            if int(data[i]) == negative_sign:
                temp.append(i)
        except:
            if int(data[i, -1]) == negative_sign:
                temp.append(i)
    return np.array(temp)


def Multi_Scale_Wavelet0(trainX, trainY, level, is_multi=True, wave_type='db1'):
    temp = [[] for i in range(level)]
    N = trainX.shape[0]
    if (is_multi == True) and (level > 1):
        for i in range(level):
            x = []
            current_level = level - i
            for _feature in range(len(trainX[0])):
                coeffs = pywt.wavedec(trainX[:, _feature], wave_type, level=level)
                for j in range(i + 1, level + 1):
                    coeffs[j] = None
                _rec = pywt.waverec(coeffs, wave_type)
                x.append(_rec[:N])

            temp[current_level - 1].extend(np.transpose(np.array(x)))

    else:
        for tab in range(level):
            current_level = level - tab
            temp[current_level - 1].extend(trainX)
    print("ALA")
    print((np.array(temp)).shape)

    return np.array(temp), trainX, trainY


def return_indexes(index_, label):
    index_1 = []
    index_2 = []
    flag = False
    index_Anomaly = index_[label == 1]  # 选出标签为1的序号
    for tab_ in range(len(index_Anomaly) - 1):
        if index_Anomaly[tab_ + 1] - index_Anomaly[tab_] > 2:  # 序号差大于2则将序号放index_2,否则放index_1
            flag = True
        if flag == True:
            try:
                index_2.append(index_Anomaly[tab_ + 1])
            except:
                pass
        else:
            index_1.append(index_Anomaly[tab_])
    # index_2.append(index_Anomaly[-1])
    index_1 = np.array(index_1)
    index_2 = np.array(index_2)
    return index_1, index_2


def multi_scale_plotting(dataA, label):
    selected_feature = 1
    original = dataA[:, selected_feature]
    plt.plot([i for i in range(len(original))], original, 'b')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Original", fontsize=12)
    plt.tick_params(labelsize=12)
    plt.grid(True)
    plt.savefig("Original.png", dpi=400)
    # plt.show()
    fig = plt.figure(figsize=(10, 5))
    label_ = np.array(label)
    # label_ = label_.astype('int64')
    # print(type(label_[0]))
    ax1 = fig.add_subplot(221)

    index_ = np.array([i for i in range(len(original))])
    index_1, index_2 = return_indexes(index_, label_)
    index_ = index_.astype('int64')
    index_1 = index_1.astype('int64')
    index_2 = index_2.astype('int64')
    plt.plot([i for i in range(len(original))], original, 'b')
    # print(label_)
    # print(label_.shape)
    # print(len(index_1))
    # print(len(original[label_ == 1]))
    # print(len(index_2))
    plt.plot(index_1, original[index_1], 'r')
    plt.plot(index_2, original[index_2], 'r')

    plt.xlabel("(a)", fontsize=10)
    plt.ylabel("Number of announced prefix")
    ax1.grid(True)
    plt.tick_params(labelsize=11)

    level_ = 2
    ax1 = fig.add_subplot(222)
    coeffs = pywt.wavedec(original, 'db1', level=level_)
    # newCoeffs = [None,coeffs[1],None]
    # new_ = pywt.waverec(newCoeffs,'db1')
    index_b = [i for i in range(len(coeffs[1]))]
    plt.plot(index_b, coeffs[1], 'b')
    # plt.plot(index_b[label_==1],new_[label_==1],'r.')
    # plt.plot([i for i in range(len(coeffs[1]))],coeffs[1],'b')
    plt.xlabel("(b)", fontsize=10)
    # plt.ylabel("Detail coefficient of level "+ str(level_))
    plt.ylabel("Number of announced prefix")
    ax1.grid(True)
    plt.tick_params(labelsize=11)

    ax1 = fig.add_subplot(223)
    level_ = 2
    coeffs = pywt.wavedec(original, 'db1', level=level_)
    newCoeffs = [None, coeffs[1], None]
    new_ = pywt.waverec(newCoeffs, 'db1')
    # print(len(index_))
    # print(len(new_))
    index_ = [i for i in range(len(new_))]
    plt.plot(index_, new_, 'b')
    plt.plot(index_1, new_[index_1], 'r')
    plt.plot(index_2, new_[index_2], 'r')
    # plt.plot([i for i in range(len(coeffs[2]))],coeffs[2],'b')
    plt.xlabel("(c)", fontsize=10)
    # plt.ylabel("Reconstructed detail of level "+str(level_))
    plt.ylabel("Number of announced prefix")
    ax1.grid(True)
    plt.tick_params(labelsize=11)

    ax1 = fig.add_subplot(224)
    level_ = 5
    coeffs = pywt.wavedec(original, 'db1', level=level_)
    newCoeffs = [coeffs[0], None, None, None, None, None]
    new_ = pywt.waverec(newCoeffs, 'db1')
    print('-------')
    print(shape(new_))
    print(new_)

    index_ = np.array([i for i in range(len(new_))])
    index_1, index_2 = return_indexes(index_, label_)
    index_ = index_.astype('int64')
    index_1 = index_1.astype('int64')
    index_2 = index_2.astype('int64')
    plt.plot(index_, new_, 'b')
    plt.plot(index_1, new_[index_1], 'r')
    plt.plot(index_2, new_[index_2], 'r')
    plt.xlabel("(d)", fontsize=10)
    # plt.ylabel("Reconstructed Approximation of level "+str(level_))
    plt.ylabel("Number of announced prefix")
    ax1.grid(True)
    plt.tick_params(labelsize=11)

    """
    ax1 = fig.add_subplot(224)
    level_ = 3
    coeffs = pywt.wavedec(original, 'db1', level=level_)
    newCoeffs = [None,coeffs[1],None,None]
    new_ = pywt.waverec(newCoeffs,'db1')
    plt.plot([i for i in range(len(new_))],new_,'b')
    #plt.plot([i for i in range(len(coeffs[3]))],coeffs[3],'b')
    plt.xlabel("(d)")
    plt.ylabel("Details of level "+str(level_))
    ax1.grid(True)
    """
    plt.tight_layout()

    plt.savefig("Wavelet Decomposition.pdf", dpi=400)
    plt.show()


def add_nosie(ratio, data):
    w = 5
    x = data[:, :-1]
    _std = x.std(axis=0, ddof=0)
    N = int(ratio * len(data))
    noise = []
    for i in range(N):
        baseinstance_index = random.randint(0, len(data) - 1)
        base_instance = data[baseinstance_index]
        noise.append([])
        for j in range(len(_std)):
            temp = random.uniform(_std[j] * -1, _std[j])
            noise[i].append(float(base_instance[j] + temp / w))
        noise[i].append(base_instance[-1])
    noise = np.array(noise)
    return np.concatenate((data, noise), axis=0)


def returnData(dataX, dataY, is_binary_class):
    global positive_sign, negative_sign
    start_ratio = 0.6
    if is_binary_class:
        positive_index = returnPositiveIndex(dataY, negative_sign)
        negative_index = returnNegativeIndex(dataY, negative_sign)
        # 对正负数据集按照，训练集，验证集，测试集6:1:3进行划分
        pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
        pos_val_index = positive_index[
                        int(start_ratio * len(positive_index)):int((start_ratio + 0.1) * len(positive_index))]
        pos_test_index = positive_index[int((start_ratio + 0.1) * len(positive_index)):len(positive_index) - 1]
        neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]
        neg_val_index = negative_index[
                        int(start_ratio * len(negative_index)):int((start_ratio + 0.1) * len(negative_index))]
        neg_test_index = negative_index[int((start_ratio + 0.1) * len(negative_index)):len(negative_index) - 1]

        train_index = np.append(neg_train_index, pos_train_index, axis=0)  # 按照原本维度合并
        train_index.sort()
        train_index = list(map(lambda a: int(a), train_index))
        train_dataX = dataX[train_index]
        train_dataY = dataY[train_index]

        val_index = np.append(neg_val_index, pos_val_index, axis=0)
        val_index.sort()
        val_index = list(map(lambda a: int(a), val_index))
        val_dataX = dataX[val_index]
        val_dataY = dataY[val_index]

        test_index = np.append(neg_test_index, pos_test_index, axis=0)
        test_index.sort()
        test_index = list(map(lambda a: int(a), test_index))
        test_dataX = dataX[test_index]
        test_dataY = dataY[test_index]
    else:
        negative_sign = 0
        negative_index = returnNegativeIndex(dataY, negative_sign)
        neg_train_index = negative_index[0:int(start_ratio * len(negative_index))]
        neg_val_index = negative_index[
                        int(start_ratio * len(negative_index)):int((start_ratio + 0.1) * len(negative_index))]
        neg_test_index = negative_index[int((start_ratio + 0.1) * len(negative_index)):len(negative_index) - 1]

        for tab_ in range(1, 5):
            negative_sign = tab_
            positive_index = returnNegativeIndex(dataY, negative_sign)

            pos_train_index = positive_index[0:int(start_ratio * len(positive_index))]
            pos_val_index = positive_index[
                            int(start_ratio * len(positive_index)):int((start_ratio + 0.1) * len(positive_index))]
            pos_test_index = positive_index[int((start_ratio + 0.1) * len(positive_index)):len(positive_index) - 1]

            train_index = np.append(neg_train_index, pos_train_index, axis=0)
            train_index.sort()
            neg_train_index = train_index

            val_index = np.append(neg_val_index, pos_val_index, axis=0)
            val_index.sort()
            neg_val_index = val_index

            test_index = np.append(neg_test_index, pos_test_index, axis=0)
            test_index.sort()
            neg_test_index = test_index

            # min_number = min(len(train_dataX),len(test_dataX))
            pprint("The training size is shapeAAA:")
            pprint("The POSITIVE TRAIN is " + str(len(pos_train_index)))
            pprint("The NEGATIVE TRAIN is " + str(len(neg_train_index)))
            pprint("The POSITIVE TEST is " + str(len(pos_test_index)))
            pprint("The NEGATIVE TEST is " + str(len(neg_test_index)))

        train_index = list(map(lambda a: int(a), train_index))
        train_dataX = dataX[train_index]
        train_dataY = dataY[train_index]

        val_index = list(map(lambda a: int(a), val_index))
        val_dataX = dataX[val_index]
        val_dataY = dataY[val_index]

        test_index = list(map(lambda a: int(a), test_index))
        test_dataX = dataX[test_index]
        test_dataY = dataY[test_index]

    # min_number = min(len(train_dataX),len(test_dataX))
    # pprint("The training size is shape:")
    # pprint(train_dataY.shape)
    # pprint("The POSITIVE is "+str(len(pos_train_index)))
    # pprint("The NEGATIVE is "+str(len(neg_train_index)))
    # pprint("The validation size is shape:")
    # pprint(val_dataX.shape)
    # pprint("The testing size is shape:")
    # pprint(test_dataX.shape)
    # pprint("The POSITIVE is "+str(len(pos_test_index)))
    # pprint("The NEGATIVE is "+str(len(neg_test_index)))

    return train_dataX, train_dataY, val_dataX, val_dataY, test_dataX, test_dataY


def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_) + 1)
    indexes = np.array(y_, dtype=np.int32)
    return np.eye(n_values)[indexes]  # Returns FLOATS


# 数值处理
def LoadData(input_data_path, filename):
    input_data_path = os.path.join(os.getcwd(), 'BGP_Data')
    with open(os.path.join(input_data_path, filename)) as fin:
        global negative_sign, positive_sign
        if filename == 'sonar.dat':
            negative_flag = 'M'
        else:
            negative_flag = '1.0'  # For binary txt flag is 1; for multi-class flag is 0
        Data = []

        for each in fin:
            if '@' in each: continue
            val = each.split(",")
            if len(val) > 0 or val[-1].strip() == "negative" or val[-1].strip() == "positive":
                if int(val[-1][0].strip()) == 1 or val[
                    -1].strip() == negative_flag:  # 1 or 1.0 as the last element of the val
                    # if val[-1].strip() != negative_flag:
                    val[-1] = negative_sign
                else:
                    val[-1] = positive_sign
                try:
                    val = list(map(lambda a: float(a), val))
                except:
                    val = list(map(lambda a: str(a), val))

                val[-1] = int(val[-1])
                Data.append(val)
        Data = np.array(Data)
        return Data


def get_data(poolingType, isNoise, noiseRatio, filePath, fileName, windowSize, trigger_flag, is_binary_class,
             multiScale=True, waveScale=-1, waveType="db1", timeScale=1):
    global positive_sign, negative_sign, output_folder
    positive_sign = 1
    negative_sign = 0
    output_folder = "output"
    # if not os.path.isdir(os.path.join(os.getcwd(), output_folder)):
    #     os.makedirs(os.path.join(os.getcwd(), output_folder))
    # data_ = LoadData(filePath, fileName)
    X, Y, data_ = swat.swat()
    scaler = preprocessing.StandardScaler()
    if isNoise == True: data_ = add_nosie(noiseRatio, data_)
    # if multiClass=="Multi":np.random.shuffle(PositiveIndex)
    if multiScale == False:
        print("AAA")
        print(windowSize)
        dataSequenlized_X, dataSequenlized_Y = slidingFunc(windowSize, scaler.fit_transform(data_[:, :-1]),
                                                           data_[:, -1])
        trainX, trainY, valX, valY, testX, testY = returnData(dataSequenlized_X, dataSequenlized_Y, is_binary_class)

    else:
        trainX_Multi = [[] for i in range(waveScale)]
        valX_Multi = [[] for i in range(waveScale)]
        testX_Multi = [[] for i in range(waveScale)]

        dataMulti, dataX, dataY = Multi_Scale_Wavelet0(data_[:, :-1], data_[:, -1], waveScale, True, waveType)

        for tab_level in range(waveScale):
            dataX_level, dataY_level = slidingFunc(windowSize, scaler.fit_transform(dataMulti[tab_level]), dataY)
            trainX, trainY, valX, valY, testX, testY = returnData(dataX_level, dataY_level, is_binary_class)

            trainX_Multi[tab_level].extend(trainX)
            valX_Multi[tab_level].extend(valX)
            testX_Multi[tab_level].extend(testX)
    if trigger_flag:
        trainY = one_hot(trainY)
        valY = one_hot(valY)
        testY = one_hot(testY)

    if multiScale == False:
        print("BBB")
        print(testX.shape)
        return trainX, trainY, valX, valY, testX, testY

    print("Multi X is ")
    print((np.array(trainX_Multi)).shape)
    print(len(trainX_Multi))
    print(len(trainX_Multi[0]))
    print(len(trainX_Multi[0][0]))

    trainX_Multi = np.array(trainX_Multi).transpose(
        (1, 0, 2, 3))  # batch_size, scale_levels, sequence_window, input_dim
    valX_Multi = np.array(valX_Multi).transpose((1, 0, 2, 3))  # batch_size, scale_levels, sequence_window, input_dim
    testX_Multi = np.array(testX_Multi).transpose((1, 0, 2, 3))  # batch_size, scale_levels, sequence_window, input_dim

    print("Input shape is" + str(trainX_Multi.shape))
    return trainX_Multi, trainY, valX_Multi, valY, testX_Multi, testY


# multi_scale_plotting_Multi(X, X)
# multi_scale_plotting(X, Y)
# res, trainX, trainY = Multi_Scale_Wavelet0(X, Y, 3, is_multi=True, wave_type='db1')
# print(res)
