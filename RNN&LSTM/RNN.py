import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from torch import nn, optim
import torch.nn.functional as F

df = pd.read_csv(r'C:\Users\roger\OneDrive\桌面\HW2\covid_19.csv')
dff = df.iloc[:,4:].values-df.iloc[:,3:-1].values
df = pd.concat([df.iloc[:,0],pd.DataFrame(dff)],axis=1)
df.index = df['Country/Region']
a = []
for i in range(df.shape[0]-1):
    for j in range(i+1,df.shape[0]):
        cor = np.corrcoef(df.iloc[i,1:].values.astype('float32'), df.iloc[j,1:].values.astype('float32'))[0,1]
        a.append([df.iloc[i,0],df.iloc[j,0],cor])
a = pd.DataFrame(a)
df1= a.pivot(index=1, columns=0, values=2)
plt.figure(figsize=(12,12))
sns.heatmap(df1.iloc[:10,:10])
a1 = a[a[2]>0.9][0].values
a2 = a[a[2]>0.9][1].values
a3 = np.append(a1,a2)
a4 = np.unique(a3)
df2 = df.loc[a4,:]
data = df2.iloc[:, 1:]
test_data_size = 20
train_data = data.iloc[:,:-test_data_size]
test_data = data.iloc[:,-test_data_size:]
#%%
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(data.shape[1]-seq_length-1):
        x = data.iloc[:,i:(i+seq_length)].values
        y = data.iloc[:,i+seq_length].values
        xs.append(x)
        yss = []
        for j in range(x.shape[0]): 
            if y[j]>0:
                yss.append(1)
            else:
                yss.append(0)
        ys.append(np.array(yss))
    return np.array(xs), np.array(ys)
seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
sample_size = train_data.shape[1]-seq_length-1
sample_size1 = test_data.shape[1]-seq_length-1
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
#%%
# =============================================================================
# RNN
# =============================================================================
X_train = X_train.reshape([sample_size,55,seq_length])
y_train = y_train.reshape([sample_size*55])
X_test = X_test.reshape([sample_size1,55,seq_length])
y_test = y_test.reshape([sample_size1*55])
class RNN(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=2)
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)   # h_state 也要作為 RNN 的一个输入

        outs = []    # 保存所有時間點的预測值
        for time_step in range(r_out.size(1)):    # 對每一個時間點計算 output
            outs.append(self.linear(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
    
def train_model(model,train_data,train_labels,test_data,test_labels):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 1000
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    h_state = None
    for t in range(num_epochs):
        y_pred ,ss= model(train_data,h_state)
        y_pred = y_pred.reshape([sample_size*55, 2])
        y_pred1 = torch.max(y_pred, 1)[1]
        accuracy = float((y_pred1 == train_labels).sum()) / float(sample_size*55)
        y = train_labels.long()
        lo = loss_fn(y_pred, y)
        with torch.no_grad():
            test_pred ,ss1= model(test_data,h_state)
            test_pred = test_pred.reshape([sample_size1*55,2])
            test_pred1 = torch.max(test_pred, 1)[1]
            accuracy1 = float((test_pred1 == test_labels).sum()) / float(sample_size1*55)
            y1 = test_labels.long()
            loss = loss_fn(test_pred, y1)
        if t%100 == 0:
            print('epoch',t,'train_loss :',lo.item(),'test_loss :',loss.item())
            print('epoch',t,'train_acc :',accuracy,'test_acc :',accuracy1)
        optimiser.zero_grad()
        lo.backward()
        optimiser.step()
        train_hist[t] = accuracy
        test_hist[t] = accuracy1
    return y_pred,train_hist,test_hist

model = RNN(
    n_features=seq_length,
    n_hidden=512,
    seq_len=55,
    n_layers=2
)
model,train,test = train_model(model, X_train, y_train, X_test, y_test)
#%%
epoch = [i for i in range(1000)]
plt.figure()
plt.plot(epoch, train)
plt.xlabel('epoch')
plt.ylabel('accuracy_rate')
plt.xticks(range(0,len(train)+1,100))
plt.title('accuracy_rate')
plt.show()   
epoch = [i for i in range(1000)]
plt.figure()
plt.plot(epoch, test)
plt.xlabel('epoch')
plt.ylabel('accuracy_rate')
plt.xticks(range(0,len(train)+1,100))
plt.title('accuracy_rate')
plt.show()   
#%%
# =============================================================================
# LSTM
# =============================================================================
X_train = X_train.reshape([sample_size,seq_length,55])
y_train = y_train.reshape([sample_size*55])
X_test = X_test.reshape([sample_size1,seq_length,55])
y_test = y_test.reshape([sample_size1*55])
class lstm(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(lstm, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=2)
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
          sequences.view(len(sequences), self.seq_len, -1),
          self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)
        y_pred = self.linear(last_time_step)
        return y_pred
    
def train_model(model,train_data,train_labels,test_data,test_labels):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 500
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(train_data).reshape([sample_size*55, 2])
        y_pred1 = torch.max(y_pred, 1)[1]
        accuracy = float((y_pred1 == train_labels).sum()) / float(sample_size*55)
        y = train_labels.long()
        lo = loss_fn(y_pred, y)
        with torch.no_grad():
            test_pred = model(test_data).reshape([sample_size1*55,2])
            test_pred1 = torch.max(test_pred, 1)[1]
            accuracy1 = float((test_pred1 == test_labels).sum()) / float(sample_size1*55)
            y1 = test_labels.long()
            loss = loss_fn(test_pred, y1)
        if t%100 == 0:
            print('epoch',t,'train_loss :',lo.item(),'test_loss :',loss.item())
            print('epoch',t,'train_acc :',accuracy,'test_acc :',accuracy1)
        optimiser.zero_grad()
        lo.backward()
        optimiser.step()
        train_hist[t] = accuracy
        test_hist[t] = accuracy1
    return y_pred,train_hist,test_hist

model = lstm(
    n_features=seq_length,
    n_hidden=512,
    seq_len=55,
    n_layers=2
)
model,train,test = train_model(model, X_train, y_train, X_test, y_test)

#%%
epoch = [i for i in range(500)]
plt.figure()
plt.plot(epoch, train)
plt.xlabel('epoch')
plt.ylabel('accuracy_rate')
plt.xticks(range(0,len(train)+1,100))
plt.title('accuracy_rate')
plt.show()   
epoch = [i for i in range(500)]
plt.figure()
plt.plot(epoch, test)
plt.xlabel('epoch')
plt.ylabel('accuracy_rate')
plt.xticks(range(0,len(train)+1,100))
plt.title('accuracy_rate')
plt.show()   
#%%
predi =  model.reshape([50,55,2])[-1]
label = torch.max(predi, 1)[1]
predi = predi.data.numpy()
predi = 1 / (1 + np.exp(-predi.max(axis = 1)))
countrys = np.array(train_data.index)
import pygal_maps_world.maps
from pygal_maps_world.maps import COUNTRIES

def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code
    return None

country_symb=[]
for i in countrys:
    country_symb.append(get_country_code(i))
al=pd.Series(predi.reshape(-1))
al.index=country_symb
al=al[al.index.notnull()]

up=al[al>=0.5]
down=al[al<0.5]
    
up=up.to_dict()
down=down.to_dict()

worldmap_chart = pygal_maps_world.maps.World()

worldmap_chart.add("ascending", up)
worldmap_chart.add("descending", down)

worldmap_chart.render_to_file('bar_chart.svg')
