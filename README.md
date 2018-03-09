**This is an automated Markdown generation from the notebook 'Crepe-Gluon.ipynb'**

# Crepe model implementation with MXNet/Gluon

This is an implementation of [the crepe mode, Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626) using the MXNet Gluon API.

We are going to perform a **text classification** task, trying to classify Amazon reviews according to the product category they belong to.

## Data download
The dataset has been made available on this website: http://jmcauley.ucsd.edu/data/amazon/, citation of relevant papers:

**Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering**
R. He, J. McAuley
*WWW*, 2016

**Image-based recommendations on styles and substitutes**
J. McAuley, C. Targett, J. Shi, A. van den Hengel
*SIGIR*, 2015




We are downloading a subset of the reviews, the k-core reviews, where k=5. That means that for each category, the dataset has been trimmed to only contain 5 reviews per individual product, and 5 reviews per user.


```python
base_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
prefix = 'reviews_'
suffix = '_5.json.gz'
folder = 'data'
categories = [
    'Home_and_Kitchen', 
    'Books', 
    'CDs_and_Vinyl', 
    'Movies_and_TV', 
    'Cell_Phones_and_Accessories',
    'Sports_and_Outdoors', 
    'Clothing_Shoes_and_Jewelry'
]
!mkdir -p $folder
for category in categories:
    print(category)
    url = base_url+prefix+category+suffix
    !wget -P $folder $url -nc
```

    Home_and_Kitchen
    File ‘data/reviews_Home_and_Kitchen_5.json.gz’ already there; not retrieving.
    
    Books
    File ‘data/reviews_Books_5.json.gz’ already there; not retrieving.
    
    CDs_and_Vinyl
    File ‘data/reviews_CDs_and_Vinyl_5.json.gz’ already there; not retrieving.
    
    Movies_and_TV
    File ‘data/reviews_Movies_and_TV_5.json.gz’ already there; not retrieving.
    
    Cell_Phones_and_Accessories
    File ‘data/reviews_Cell_Phones_and_Accessories_5.json.gz’ already there; not retrieving.
    
    Sports_and_Outdoors
    File ‘data/reviews_Sports_and_Outdoors_5.json.gz’ already there; not retrieving.
    
    Clothing_Shoes_and_Jewelry
    File ‘data/reviews_Clothing_Shoes_and_Jewelry_5.json.gz’ already there; not retrieving.
    


## Data Pre-processing
We need to perform some pre-processing steps in order to have the data in a format we can use for training (**X**,**Y**)
In order to speed up training and balance the dataset we will only use a subset of reviews for each category.

### Load the data in memory


```python
MAX_ITEMS_PER_CATEGORY = 250000
```

Helper functions to read from the .json.gzip files


```python
import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for line in g:
        yield eval(line)

def get_dataframe(path, num_lines):
    i = 0
    df = {}
    for d in parse(path):
        if i > num_lines:
            break
        df[i] = d
        i += 1

    return pd.DataFrame.from_dict(df, orient='index')
```

    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/matplotlib/__init__.py:962: UserWarning: Duplicate key in file "/home/ec2-user/.config/matplotlib/matplotlibrc", line #2
      (fname, cnt))
    /home/ec2-user/anaconda3/envs/mxnet_p36/lib/python3.6/site-packages/matplotlib/__init__.py:962: UserWarning: Duplicate key in file "/home/ec2-user/.config/matplotlib/matplotlibrc", line #3
      (fname, cnt))


For each category we load MAX_ITEMS_PER_CATEGORY by randomly sampling the files and shuffling


```python
# Loading data from file if exist
try:
    data = pd.read_pickle('pickleddata.pkl')
except:
    data = None
```


```python
if data is None:
    data = pd.DataFrame(data={'X':[],'Y':[]})
    for index, category in enumerate(categories):
        df = get_dataframe("{}/{}{}{}".format(folder, prefix, category, suffix), MAX_ITEMS_PER_CATEGORY)    
        # Each review's summary is prepended to the main review text
        df = pd.DataFrame(data={'X':(df['summary']+' | '+df['reviewText'])[:MAX_ITEMS_PER_CATEGORY],'Y':index})
        data = data.append(df)
        print('{}:{} reviews'.format(category, len(df)))

    # Shuffle the samples
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    # Saving the data in a pickled file
    pd.to_pickle(data, 'pickleddata.pkl')
```

Let's visualize the data:


```python
print(data['Y'].value_counts())
data.head()
```

    1.0    250000
    6.0    250000
    5.0    250000
    3.0    250000
    2.0    250000
    0.0    250000
    4.0    194439
    Name: Y, dtype: int64





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Love these! | Love these for microwave use and...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Unholy Trinity of 1986, Part 1.  It's rain...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>My favorite Disney Movie! | I love the scene w...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dashboard Universal Car Mount holder - Perfect...</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Great :) | Great for crop tops and other belly...</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the dataset


```python
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.data import ArrayDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import multiprocessing
```


```python
ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)} # { a: 0, b: 1, etc}
FEATURE_LEN = 1014 # max-length in characters for one document
BATCH_SIZE = 128 # number of documents per batch
NUM_FILTERS = 256 # number of convolutional filters per convolutional layer
NUM_OUTPUTS = len(categories) # number of classes
FULLY_CONNECTED = 1024 # number of unit in the fully connected dense layer
DROPOUT_RATE = 0.5 # probability of node drop out
LEARNING_RATE = 0.01 # learning rate of the gradient
MOMENTUM = 0.9 # momentum of the gradient
WDECAY = 0.00001 # regularization term to limit size of weights
NUM_WORKERS = multiprocessing.cpu_count() # number of workers used in the data loading
```


```python
def encode(text):
    encoded = np.zeros([len(ALPHABET), FEATURE_LEN], dtype='float32')
    review = text.lower()[::-1]
    i = 0
    for letter in text:
        if i >= FEATURE_LEN:
            break;
        if letter in ALPHABET_INDEX:
            encoded[ALPHABET_INDEX[letter]][i] = 1
        i += 1
    return encoded
```


```python
class AmazonDataSet(ArrayDataset):
    # We pre-process the documents on the fly
    def __getitem__(self, idx):
        return encode(self._data[0][idx]), self._data[1][idx]
        
```


```python
split = 0.8
split_index = int(split*len(data)/BATCH_SIZE)*BATCH_SIZE
train_dataset = AmazonDataSet(data['X'][:split_index].as_matrix(),data['Y'][:split_index].as_matrix())
test_dataset = AmazonDataSet(data['X'][split_index:].as_matrix(),data['Y'][split_index:].as_matrix())
```


```python
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
```


```python
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE*6, num_workers=NUM_WORKERS*2)
```

## Creation of the network


```python
ctx = mx.gpu() # use ctx = mx.cpu() to run on CPU
```


```python
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
    net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'))
    net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
    net.add(gluon.nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'))
    net.add(gluon.nn.MaxPool1D(pool_size=3, strides=3))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
    net.add(gluon.nn.Dropout(DROPOUT_RATE))
    net.add(gluon.nn.Dense(FULLY_CONNECTED, activation='relu'))
    net.add(gluon.nn.Dropout(DROPOUT_RATE))
    net.add(gluon.nn.Dense(NUM_OUTPUTS))

```


```python
print(net)
```

    HybridSequential(
      (0): Conv1D(None -> 256, kernel_size=(7,), stride=(1,))
      (1): MaxPool1D(size=(3,), stride=(3,), padding=(0,), ceil_mode=False)
      (2): Conv1D(None -> 256, kernel_size=(7,), stride=(1,))
      (3): MaxPool1D(size=(3,), stride=(3,), padding=(0,), ceil_mode=False)
      (4): Conv1D(None -> 256, kernel_size=(3,), stride=(1,))
      (5): Conv1D(None -> 256, kernel_size=(3,), stride=(1,))
      (6): Conv1D(None -> 256, kernel_size=(3,), stride=(1,))
      (7): Conv1D(None -> 256, kernel_size=(3,), stride=(1,))
      (8): MaxPool1D(size=(3,), stride=(3,), padding=(0,), ceil_mode=False)
      (9): Flatten
      (10): Dense(None -> 1024, Activation(relu))
      (11): Dropout(p = 0.5)
      (12): Dense(None -> 1024, Activation(relu))
      (13): Dropout(p = 0.5)
      (14): Dense(None -> 7, linear)
    )



```python
hybridize = True # for speed improvement, compile the network but no in-depth debugging possible
load_params = True # Load pre-trained model
```

### Parameter initialization


```python
if load_params:
    net.load_params('crepe_gluon_epoch6.params', ctx=ctx)
else:
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

### Hybridization


```python
if hybridize:
    net.hybridize()
```

### Softmax cross-entropy Loss


```python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

### Optimizer


```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', 
                        {'learning_rate': LEARNING_RATE, 
                         'wd':WDECAY, 
                         'momentum':MOMENTUM})
```

### Evaluate Accuracy


```python
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        prediction = nd.argmax(output, axis=1)

        if (i%50 == 0):
            print("Samples {}".formaat(i*len(data)))
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]
```

### Training Loop


```python
start_epoch = 6
number_epochs = 7
smoothing_constant = .01
for e in range(start_epoch, number_epochs):
    for i, (review, label) in enumerate(train_dataloader):
        review = review.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(review)
        with autograd.record():
            output = net(review)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(review.shape[0])
        
        # moving average of the loss
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if (i == 0) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        if (i%50 == 0):
            nd.waitall()
            print('Batch {}:{},{}'.format(i,curr_loss,moving_loss))

    test_accuracy = evaluate_accuracy(test_dataloader, net)
    #Save the model using the gluon params format
    net.save_params('crepe_epoch_{}_test_acc_{}.params'.format(e,int(test_accuracy*10000)/100))
    print("Epoch %s. Loss: %s, Test_acc %s" % (e, moving_loss, test_accuracy))
```

    Batch 0:0.15787965059280396,0.15787965059280396
    Batch 50:0.10220436751842499,0.15457812894300896
    Batch 100:0.17303970456123352,0.15119215689855275
    Batch 150:0.2697896361351013,0.1522699015862132
    Batch 200:0.21811547875404358,0.1475157505321016
    Batch 250:0.09753967821598053,0.14239624025142694
    Batch 300:0.14830192923545837,0.14575528771982066
    Batch 350:0.18873190879821777,0.15026100733316544
    Batch 400:0.11522325873374939,0.14915754813674642
    Batch 450:0.15397216379642487,0.14515570012827422
    Batch 500:0.12619560956954956,0.13938639690421473
    Batch 0
    Batch 50
    Batch 100
    Batch 150
    Batch 200
    Batch 250
    Batch 300
    Batch 350
    Batch 400
    Epoch 6. Loss: 0.139386396904, Test_acc 0.932054561709



```python
# clear the shared memory
!rm -rf /dev/shm/*
```

### Export to the symbolic format


```python
#net.export('model/crepe')
```

### Random testing


```python
import random
index = random.randint(1, len(data))
review = data['X'][index]
label = categories[int(data['Y'][index])]
print(review)
print('Category: {}'.format(label))
encoded = nd.array([encode(review)], ctx=ctx)
output = net(encoded)
predicted = categories[np.argmax(output[0].asnumpy())]
if predicted == label:
      print('Correct')
else:
      print('Incorrectly predicted {}'.format(predicted))
```

    Great Phones, great price! | I like these headphones a lot. The service was fast, the product came exactly when expected.Overall one does have to get used to the noise cancellation aspect of these phones, because it really does work. Aside from the adjustment to these, they work fantastic!
    Category: Cell_Phones_and_Accessories
    Correct


### Manual Testing


```python
review_title = "Good stuff"
review = "This album is definitely above the previous one"
```


```python
print(review_title)
print(review + '\n')
encoded = nd.array([encode(review + " | " + review_title)], ctx=ctx)
output = net(encoded)
softmax = nd.exp(output) / nd.sum(nd.exp(output))[0]
predicted = categories[np.argmax(output[0].asnumpy())]
print('Predicted: {}'.format(predicted))
for i, val in enumerate(categories):
    print(val, float(int(softmax[0][i].asnumpy()*1000)/10), '%')
```

    Good stuff
    This album is definitely above the previous one
    
    Predicted: CDs_and_Vinyl
    Home_and_Kitchen 0.0 %
    Books 0.0 %
    CDs_and_Vinyl 99.7 %
    Movies_and_TV 0.1 %
    Cell_Phones_and_Accessories 0.0 %
    Sports_and_Outdoors 0.0 %
    Clothing_Shoes_and_Jewelry 0.0 %

