# 循环神经网络

## 一、introduction
在处理序列数据时，我们希望能根据已经有的序列数据来生成后续的数据，可以有两种实现思路：

1. n元语法模型
2. 隐变量模型

前者在t时刻生成单词$x_t$,仅依赖于前面$n-1$个单词，而后者依赖于前$t-1$个时刻所生成的隐藏状态$h_{t-1}$。

二者由明显的优劣对比，当序列长度增加时，n元语法模型所需要存储的单词表的规模是以指数增加，因此后续主要讨论隐变量模型。

---
### 隐变量模型
对于一个足够强大的隐变量模型而言，$h_{t-1}$应该是迄今为止观察到的全部数据的抽象。而循环神经网络RNN是具有隐藏状态的神经网络。

### 无隐藏状态的神经网络（有举例）
即一般的多层感知机，由隐藏层但无隐藏状态，隐藏层的输出为输入tensor经过线性运算和激活函数后的结果。

#### coding：

规定相关长度为4，以加噪的正弦为训练数据。训练时全部使用真实的序列，即使用4个groundtruth来生成一个数据。

- 验证模型的单步预测能力
- 验证模型的多步预测能力



# 二、循环神经网络
相比于一般的神经网络，在激活函数前多了一项$H_{t-1}$的加权项。
$$
\mathbf{H}_t=\phi\left(\mathbf{X}_t \mathbf{W}_{x h}+\mathbf{H}_{t-1} \mathbf{W}_{h h}+\mathbf{b}_h\right)
$$

![image-20220926224254255](C:\Users\MrXu\AppData\Roaming\Typora\typora-user-images\image-20220926224254255.png)

![image-20220926224322051](C:\Users\MrXu\AppData\Roaming\Typora\typora-user-images\image-20220926224322051.png)

在每个时间步生成一个label。



## 循环神经网络的实现

### 独热码

用来编码输入数据，独热码向量的长度与词元类别数有关。

### 数据维度

每次采样到的数据为（batch_size, num_timestep），经过one-hot编码后，数据维度是（batch_size, num_timestep, len(vocab)），为了一步步更新，通常转换输入数据维度为（num_timestep, batch_size, len(vocab)）。 

**backpropagation through time**

### 初始化模型参数

```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### 初始化第一个隐藏状态

```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

最开始没有来自之前的隐藏状态，因此返回0.

### RNN核心代码

```python
def rnn(inputs, state, params):
    # `inputs`的形状：(`时间步数量`，`批量大小`，`词表大小`)
    W_xh, W_hh, b_h, W_hq, b_q = params # 这里的参数都是可学习的，在之前已经设定了
    H, = state # H表示上一时刻的状态
    outputs = []
    # `X`的形状：(`批量大小`，`词表大小`)
    for X in inputs:  # 即对于每一个时间步
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 计算新的H
        Y = torch.mm(H, W_hq) + b_q  # 计算当前时间步的输出，形状为(`批量大小`，`词表大小`)
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

**使用类封装**

```python
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)  # 随机初始化网络模型参数
        self.init_state, self.forward_fn = init_state, forward_fn  # 得到初始化状态和前向传播函数

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # 注意此处的转置，将时间步放到了第一个维度
        return self.forward_fn(X, state, self.params)  # 每次执行自动调用传入的前向传播函数

    def begin_state(self, batch_size, device):  # 最初的状态
        return self.init_state(batch_size, self.num_hiddens, device)
```

**设定超参数初始化模型**

```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
```

**训练RNN**

```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期（定义见第8章）。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和, 词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化`state`
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state`对于`nn.GRU`是个张量
                state.detach_()
            else:
                # `state`对于`nn.LSTM`或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了`mean`函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

**推理**

```python
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在`prefix`后面生成新字符。"""
    state = net.begin_state(batch_size=1, device=device)  # 随机初始化
    outputs = [vocab[prefix[0]]] 
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)  # 让state不断更新，不关心输出，输出用groundtruth
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测`num_preds`步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 取最大的作为预测输出词元，实质是分类问题
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

**评价**

梯度消失和梯度爆炸问题严重

## 三、RNN的发展

- LSTM
- GRU