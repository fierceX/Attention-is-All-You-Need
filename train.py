#%%
from collections import Counter
import os
import random
import time
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from tqdm import tqdm
from layer import *


#%%
def readIMDB(dir_url, seg='train'):
    pos_or_neg = ['pos', 'neg']
    dataset = []
    for lb in pos_or_neg:
        files = os.listdir(dir_url + '/' + seg + '/' + lb + '/')
        for file in tqdm(files):
            with open(
                    dir_url + '/' + seg + '/' + lb + '/' + file,
                    'r',
                    encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if lb == 'pos':
                    dataset.append([review, 1])
                elif lb == 'neg':
                    dataset.append([review, 0])
    return dataset


#%%
data_dir = './aclImdb/'
context = mx.gpu(1)
batch_size = 32
max_len = 100
embeding_size = 128

#%%
train_dataset = readIMDB(data_dir, 'train')
test_dataset = readIMDB(data_dir, 'test')

# shuffle 数据集。
random.shuffle(train_dataset)
random.shuffle(test_dataset)


#%%
def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]


#%%
train_tokenized = []
train_labels = []
for review, score in train_dataset:
    train_tokenized.append(tokenizer(review))
    train_labels.append(score)
test_tokenized = []
test_labels = []
for review, score in test_dataset:
    test_tokenized.append(tokenizer(review))
    test_labels.append(score)

#%%
token_counter = Counter()


def count_token(train_tokenized):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1


count_token(train_tokenized)
vocab = text.vocab.Vocabulary(
    token_counter, unknown_token='<unk>', reserved_tokens=None)


#%%
# 根据词典，将数据转换成特征向量。
def encode_samples(x_raw_samples, vocab):
    x_encoded_samples = []
    for sample in x_raw_samples:
        x_encoded_sample = []
        for token in sample:
            if token in vocab.token_to_idx:
                x_encoded_sample.append(vocab.token_to_idx[token])
            else:
                x_encoded_sample.append(0)
        x_encoded_samples.append(x_encoded_sample)
    return x_encoded_samples


# 将特征向量补成定长。
def pad_samples(x_encoded_samples, maxlen=100, val=0):
    x_samples = []
    for sample in x_encoded_samples:
        if len(sample) > maxlen:
            new_sample = sample[:maxlen]
        else:
            num_padding = maxlen - len(sample)
            new_sample = sample
            for i in range(num_padding):
                new_sample.append(val)
        x_samples.append(new_sample)
    return x_samples


#%%
x_encoded_train = encode_samples(train_tokenized, vocab)
x_encoded_test = encode_samples(test_tokenized, vocab)

x_train = nd.array(pad_samples(x_encoded_train, max_len, 0), ctx=context)
x_test = nd.array(pad_samples(x_encoded_test, max_len, 0), ctx=context)

y_train = nd.array([score for text, score in train_dataset], ctx=context)
y_test = nd.array([score for text, score in test_dataset], ctx=context)

#%%
train_data = gluon.data.ArrayDataset(x_train, y_train)
train_dataloader = gluon.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)

test_data = gluon.data.ArrayDataset(x_test, y_test)
test_dataloader = gluon.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True)


#%%
def eval(dataloader):
    total_L = 0
    ntotal = 0
    accuracy = mx.metric.Accuracy()
    for data, label in dataloader:
        output = net(data)
        label = label.as_in_context(context)
        L = loss(output, label)
        total_L += nd.sum(L).asscalar()
        ntotal += L.size
        predicts = nd.argmax(output, axis=1)
        accuracy.update(preds=predicts, labels=label)
    return total_L / ntotal, accuracy.get()[1]


#%%
net = SANet(
    shape=(embeding_size, max_len), Vocad_len=len(vocab), h=8, Is_PE=False)
net.initialize(mx.init.Uniform(.1), ctx=context)
net.hybridize()
trainer = gluon.Trainer(net.collect_params(), 'Adam')
loss = gluon.loss.SoftmaxCrossEntropyLoss()
loss.hybridize()

#%%
num_epochs = 2
start_train_time = time.time()
for epoch in range(num_epochs):
    start_epoch_time = time.time()
    total_L = 0
    ntotal = 0

    for i, (data, label) in enumerate(train_dataloader):
        with autograd.record():
            output = net(data)
            L = loss(output, label.as_in_context(context))
        L.backward()
        trainer.step(batch_size)
        total_L += nd.sum(L).asscalar()
        ntotal += L.size
        if i % 30 == 0 and i != 0:
            print('Epoch %d. batch %d. Loss %6f' % (epoch, i,
                                                    total_L / ntotal))
            total_L = 0
            ntotal = 0

    print('performing testing:')
    train_loss, train_acc = eval(train_dataloader)
    test_loss, test_acc = eval(test_dataloader)

    print('[epoch %d] train loss %.6f, train accuracy %.2f' %
          (epoch, train_loss, train_acc))
    print('[epoch %d] test loss %.6f, test accuracy %.2f' % (epoch, test_loss,
                                                             test_acc))
    print('[epoch %d] throughput %.2f samples/s' %
          (epoch,
           (batch_size * len(x_train)) / (time.time() - start_epoch_time)))
    print('[epoch %d] total time %.2f s' % (epoch,
                                            (time.time() - start_epoch_time)))

print('total training throughput %.2f samples/s' %
      ((batch_size * len(x_train) * num_epochs) /
       (time.time() - start_train_time)))
print('total training time %.2f s' % ((time.time() - start_train_time)))