#%%
# =============================================================================
# 1.前處理
# =============================================================================
import pandas as pd
from PIL import Image
import numpy as np

class Image_preprocessing(object):
    def __init__(self, data, xsize, ysize): #xsize, ysize想要的像素
        self.xsize = xsize
        self.ysize = ysize
        self.data = data
    def get_face(self):
        im3 = []
        im = Image.open(path +'/'+ self.filename)
        im = im.convert('RGB')
        im1 = np.asarray(im, dtype = np.uint8)
        d1 = self.data[self.data['filename']== self.filename].reset_index()
        for i in range(d1.shape[0]):
            xmin = d1['xmin'][i]
            xmax = d1['xmax'][i]
            ymin = d1['ymin'][i]
            ymax = d1['ymax'][i]
            im2 = im1[ymin:ymax,xmin:xmax,]
            im4 = Image.fromarray(im2,'RGB').resize((self.xsize,self.ysize))
            im3.append(np.asarray(im4, dtype = np.uint8))
        return im3
    def get_data(self):
        files = self.data['filename'].unique()
        faces = []
        for j in range(len(files)):
            self.filename = files[j]
            faces += self.get_face()
        return np.array(faces).reshape([len(faces),self.xsize*self.ysize,3])

train = pd.read_csv(r'C:\Users\517super\Desktop\Deep_Learning\CNN\train.csv')
test = pd.read_csv(r'C:\Users\517super\Desktop\Deep_Learning\CNN\test.csv')
path = r'C:\Users\517super\Desktop\Deep_Learning\CNN\images'
xtrain = Image_preprocessing(train,32,32).get_data()   
xtest = Image_preprocessing(test,32,32).get_data()    
mapp = {'good':2,'none':1,'bad':0}
train['label'] = train['label'].map(mapp)
test['label'] = test['label'].map(mapp)
ytrain = train['label'].values    
ytest = test['label'].values
def one_hot(df):
    a = df.size
    y_one_hot = np.zeros([a, np.amax(df)+1])
    y_one_hot[np.arange(a), df.reshape(1, df.size)]  = 1
    return(y_one_hot)
ytrain = one_hot(ytrain)   
ytest = one_hot(ytest)
xtrain = xtrain/255
xtest = xtest/255
#%%    
# =============================================================================
# CNN
# =============================================================================

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

learning_rate = 0.01

# Network Parameters
n_input = 1024*3 
n_classes = 3
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 1024, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
def Weight(shape):
    st = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(st)

def Bias(shape):
    st = tf.constant(0.1, shape = shape)
    return tf.Variable(st)
# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1],padding='VALID'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img,ksize=[1, k, k, 1],strides=[1, k, k, 1],padding='SAME')

# Store layers weight & bias

wc1 = Weight([5, 5, 3, 32]) # 第1個卷積5x5 conv, 3 input, 32 outputs
wc2 = Weight([5, 5, 32, 64]) # 第2個卷積5x5 conv, 32 inputs, 64 outputs
wd1 = Weight([1600, 1024]) # fully connected, 1600 inputs, 1024 outputs
wout =Weight([1024, n_classes]) # 1024 inputs, 3 outputs (class prediction)
bc1 = Bias([32])
bc2 = Bias([64])
bd1 = Bias([1024])
bout = Bias([n_classes])
# Construct model
_X = tf.reshape(x, shape=[-1, 32,32, 3])
# Convolution Layer
conv1 = tf.nn.relu(conv2d(_X,wc1,bc1))
# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)
# Convolution Layer
conv2 = tf.nn.relu(conv2d(conv1,wc2,bc2))
# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)
# Apply Dropout
conv2 = tf.nn.dropout(conv2, keep_prob)
# Fully connected layer
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) # Relu activation
dense1 = tf.nn.dropout(dense1, keep_prob) # Apply Dropout
# Output, class prediction
pred = tf.matmul(dense1, wout)+ bout
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
batch = 256
result = []
with tf.Session() as sess:
    sess.run(init)
    step = 1
    num = 0
    # Keep training until reach max iterations
    for i in range(100):
        if num+batch >= xtrain.shape[0]:
            num1 = xtrain.shape[0]
        else :
            num1 = num+batch
        batch_train = xtrain[num:num1,:,:]
        batch_l = ytrain[num:num1,:]
        # Fit training using batch data
        optimizer.run(feed_dict={x: batch_train, y: batch_l, keep_prob: dropout})
        pre = pred.eval(feed_dict={x: xtrain, y: ytrain, keep_prob: 1.})
        acc1 = accuracy.eval(feed_dict={x: xtrain, y: ytrain, keep_prob: 1.})
        loss = cost.eval(feed_dict={x: xtrain, y: ytrain, keep_prob: 1.})
        acc2 = accuracy.eval(feed_dict={x: xtest, y: ytest, keep_prob: 1.})
        print ('step',i,"Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc1) + ", Testing Accuracy= " + "{:.5f}".format(acc2))
        result.append([loss,acc1,acc2])
        step += 1
        if num1 == xtrain.shape[0]:
            num = 0
        else:
            num = num1
    print ("Optimization Finished!")
    result1 = pred.eval(feed_dict={x: xtrain, y: ytrain, keep_prob: 1.})
    result2 = pred.eval(feed_dict={x: xtest, y: ytest, keep_prob: 1.})
#%%
import matplotlib.pyplot as plt
result = np.array(result)
j = list(range(50))
j1 = ['Average Cross Entropy','Training Accuracy','Testing Accuracy']
plt.plot(j, result[:,0])
plt.xlabel('Number of epoch')
plt.ylabel('Average Cross Entropy')
#plt.savefig("Average_Cross_Entropy.png")
plt.show()  
for i in range(1,3):
    plt.plot(j, result[:,i])
    plt.xlabel('Number of epoch')
    plt.ylabel('Accuracy') 
#plt.savefig("Accuracy.png")
plt.show()   

#%%
res = result1.argmax(1)
lab = train['label'].values
res1 = result2.argmax(1)
lab1 = test['label'].values
ac = []
for i in range(3):
    ac.append([len(res[lab==i][res[lab==i]==i])/len(lab[lab==i]),len(res1[lab1==i][res1[lab1==i]==i])/len(lab1[lab1==i])])
ac_rate = pd.DataFrame(ac,columns = ['Training Accuracy','Testing Accuracy'])
#ac_rate.to_excel('ac_rate.xlsx')

#%%
from PIL import ImageDraw,ImageFont
pic = '0_10725.jpg'
pic_train = train[train['filename']==pic]
ind = np.array(pic_train.index)
im =Image.open(path+"/"+pic )
draw =ImageDraw.Draw(im)
font  = ImageFont.truetype("arial.ttf", 15)
pic_pred = res[ind]
pic_pred1 = []
for j in range(len(pic_pred)):
    if pic_pred[j]==2:
        pic_pred1.append('Good')
    elif pic_pred[j]==1:
        pic_pred1.append('None')
    else:
        pic_pred1.append('Bad')
for i in ind:
    if res[i]==2:
        color=(0,255,0)
        draw.rectangle((pic_train['xmin'][i],pic_train['ymax'][i],pic_train['xmin'][i]+45,pic_train['ymax'][i]+20),fill= color,width = 1)
    elif res[i]==0:
        color=(255,0,0)
        draw.rectangle((pic_train['xmin'][i],pic_train['ymax'][i],pic_train['xmin'][i]+45,pic_train['ymax'][i]+20),fill= color,width = 1)
    else :
        color=(0,0,255)
        draw.rectangle((pic_train['xmin'][i],pic_train['ymax'][i],pic_train['xmin'][i]+45,pic_train['ymax'][i]+20),fill= color,width = 1)
    draw.rectangle((pic_train['xmin'][i],pic_train['ymin'][i],pic_train['xmax'][i],pic_train['ymax'][i]),outline= color,width = 4)
    draw.text((pic_train['xmin'][i],pic_train['ymax'][i])," "+pic_pred1[i-ind[0]], fill = (0,0,0),font=font)
im.show()
im.save("2-3.png")


