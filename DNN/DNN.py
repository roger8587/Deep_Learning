#%%
import numpy as np
np.random.seed(0)
class Activator(object):
    def forward(self, weighted_input): #前向傳播計算輸出
        return 1.0 / (1.0 + np.exp(-weighted_input))
    def backward(self, output):  #后向傳播計算導數
        return np.multiply(output,(1 - output))   # 對應元素相乘
        return output
# 全連接每層的實現類。輸入對象x、神經層輸出a、輸出y均為列向量
class FullConnectedLayer(object):
    # 構造函數。input_size: 本層輸入向量的維度。output_size: 本層輸出向量的維度。activator: 激活函數
    def __init__(self, input_size, output_size,activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 權重數組W
        self.W = np.random.uniform(-0.1, 0.1,(output_size, input_size))  #初始化為-0.1~0.1之間的數。權重的大小。行數=輸出個數，列數=輸入個數。a=w*x，a和x都是列向量
        #self.W = np.zeros((output_size, input_size)) #初始權重為0向量
        # 偏置項b
        self.b = np.zeros((output_size, 1))  # 全0列向量偏重項
        # 輸出向量
        self.output = np.zeros((output_size, 1)) #初始化為全0列向量

    # 前向計算，預測輸出。input_array: 輸入向量，維度必須等於input_size
    def forward(self, input_array):   # 式2
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    # 反向計算W和b的梯度。delta_array: 從上一層傳遞過來的誤差項。列向量
    def backward(self, delta_array):
        # 式8
        self.delta = np.multiply(self.activator.backward(self.input),np.dot(self.W.T, delta_array))   #計算當前層的誤差，已被上一層使用
        self.W_grad = np.dot(delta_array, self.input.T)   # 計算w的梯度。梯度=誤差.*輸入
        self.b_grad = delta_array  #計算b的梯度

    # 使用梯度下降算法更新權重
    def update(self, learning_rate):
        self.W -= learning_rate * self.W_grad
        self.b -= learning_rate * self.b_grad


# 神經網絡類
class Network(object):
    # 初始化一個全連接神經網絡。layers:數組，描述神經網絡每層節點數。包含輸入層節點個數、隱藏層節點個數、輸出層節點個數
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i+1],Activator()))   # 創建全連接層，並添加到layers中
    # 訓練函數。labels: 樣本標簽矩陣。data_set: 輸入樣本矩陣。rate: 學習速率。epoch: 訓練輪數
    def train(self, xlabels, xdata_set, ylabels, ydata_set, rate, epoch):
        result = []
        for i in range(epoch):
            noupt = []
            cre = []
            for d in range(len(xdata_set)):
                oneobject = np.array(xdata_set[d]).reshape(-1,1)   #將輸入對象和輸出標簽轉化為列向量
                self.onelabel = np.array(xlabels[d]).reshape(-1,1)
                self.train_one_sample(self.onelabel,oneobject, rate)
                cre.append(self.cross_entropy(xlabels[d]))
                noupt.append(self.layers[-2].output)
            loss = np.sum(np.array(cre))/len(xdata_set) 
            train_accuracy = self.accuracy_rate(xdata_set,xlabels)[0]
            test_accuracy = self.accuracy_rate(ydata_set,ylabels)[0]
            print('[Epoch %d]: Training Loss：%0.4f, Training Accuracy : %0.2f%%, Test Accuracy : %0.2f%%' % (i+1,loss,train_accuracy*100,test_accuracy*100))# 內部函數，用一個樣本訓練網絡
            result.append([loss,train_accuracy,test_accuracy])
        return(result,np.array(noupt))
    def train_one_sample(self, label, sample, rate):
        # print('樣本：\n',sample)
        self.predict(sample)  # 根據樣本對象預測值
        self.calc_gradient(label) # 計算梯度
        self.update_weight(rate) # 更新權重

    # 使用神經網絡實現預測。sample: 輸入樣本
    def predict(self, sample):
        sample = sample.reshape(-1,1)   #將樣本轉換為列向量
        output = sample  # 輸入樣本作為輸入層的輸出
        for layer in self.layers:
            layer.forward(output)  # 逐層向后計算預測值。因為每層都是線性回歸
            output = layer.output
            
        return output

    # 計算每個節點的誤差。label為一個樣本的輸出向量，也就對應了最后一個所有輸出節點輸出的值
    def calc_gradient(self, label):
        # print('計算梯度：',self.layers[-1].activator.backward(self.layers[-1].output).shape)
        delta = np.multiply(self.layers[-1].activator.backward(self.layers[-1].output),(self.layers[-1].output - label))  #計算輸出誤差
        # print('輸出誤差：', delta.shape)
        for layer in self.layers[::-1]:
            layer.backward(delta)   # 逐層向前計算誤差。計算神經網絡層和輸入層誤差
            delta = layer.delta
            # print('當前層誤差：', delta.shape)
        return delta

    # 更新每個連接權重
    def update_weight(self, rate):
        for layer in self.layers:  # 逐層更新權重
            layer.update(rate)
    def cross_entropy(self,label):
        sm = softmax(self.layers[-1].output)
        en1 = np.log(sm)
        return(-label.dot(en1))
    def accuracy_rate(self,data_set,labels):
        total = data_set.shape[0]
        acc = 0
        matrix = np.zeros((10,10))
        for i in range(total):
            predict1 = self.predict(data_set[i,:])
            predict = tclass(predict1)
            label = tclass(labels[i])
            if label == predict:
                acc += 1
            matrix[label,predict] += 1
        return([float(acc) / float(total),matrix])
# 根據返回結果計算所屬類型
def tclass(vec):
    return vec.argmax(axis=0)   # 獲取概率最大的分類，由於vec是列向量，所以這里按列取最大的位置
def softmax(x):
    return np.exp(x) / np.exp(x).sum()