# Transformer全解析

## 前言

**此文档是我本周paper reading 任务的一个总结，基于Google在2017年的论文《Attention is all you need》，代码部分的解析基于Pytorch（众所周知的原因，原文给出的是Tensorflow）**

## Transformer是解决什么问题的

在序列处理任务中（例如机器翻译），常用的是RNN及其衍生模型，他们能对序列任务进行建模，但是存在一些缺点：

- 梯度消失/梯度爆炸（RNN）
- 长距离建模能力差
- 不能并行计算，一次一个时间步

Transformer从结构层面改善了这些问题，基于Attention机制，拥有并行计算能力，且在多种任务上具有优秀的表现。

## 模型架构

有句话这么说：任何太过先进的科技对普通人而言就像魔法。Transformer的神奇之处在哪里，我们来看他的模型架构：

![](https://s1.ax1x.com/2020/04/25/JyCdy9.png#shadow)

可见明显的编码器-解码器架构,输入从编码器进来，得到隐藏表示，从编码器输出；解码器先获取0到当前时刻全部的序列输出，通过自注意力编码后送到解码器，融合来自编码器的信息后得到输出。

接下来展开说说模型架构中各个部分的功能和数据维度的变化：

- 输入是什么样的？

  如果是对于翻译任务，往往输入的一个样本就是一个句子，维度是```(batch_size, num_steps)```，批量大小不难理解，就是有几个句子，num_steps代表一个句子有多少个Token。什么是Token？就是词元（可能是一个单词，或者一个短语）。
  
- Input Embedding

  输入的Token是标量，是单个的数字编号，而训练所需的是Tensor，Embedding层通过可学习的参数编码，将一个标量编码成一个embedding向量，数据维度变成了```(batch_size, num_steps, embedding_dim)```
  
- Positional encoding

  Transformer基于注意力机制进行并行的计算，因此无法像RNN一样隐式得给出序列中各个词元的位置信息，因此需要专门加上位置信息的表示模块。

  具体的做法是使用正弦函数编码
  $$
  PE{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}}) \\ PE{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
  $$
  pos是每个词元的位置，i是embedding的维度数，最大不超过embedding_dim的一半。为什么是一半？因为以上两个公式，每个i都只需要取一半就可以走完整个序列。
  
  编码后positional encoding维度：```(batch_size, num_steps, embedding_dim)```
  
  编码效果图：
  
  ![](https://s1.ax1x.com/2020/04/25/JyRShD.png#shadow)
  
  实现代码
  
  ```python
  class PositionalEncoding(nn.Module):
      def __init__(self, d_model, dropout=0.1, max_len=5000):  # 最大的序列长度，位置编码必须能够覆盖
          super(PositionalEncoding, self).__init__()
          self.dropout = nn.Dropout(p=dropout)  # 编码后做一个dropout
  
          pe = torch.zeros(max_len, d_model)  # 先创建出位置编码的数据维度
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
          div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
          pe[:, 0::2] = torch.sin(position * div_term)
          pe[:, 1::2] = torch.cos(position * div_term)
          pe = pe.unsqueeze(0).transpose(0, 1)
          self.register_buffer('pe', pe)
  
      def forward(self, x):
          '''
          x: [seq_len, batch_size, d_model]
          '''
          x = x + self.pe[:x.size(0), :]
          return self.dropout(x)
  ```
  
  **显然，Input Embedding和Positional encoding具有相同的维度，二者可以直接相加，现在的数据维度是：**```(batch_size, num_steps, embedding_dim)```

- Encoder module

  编码器是包括多个编码器模块的，在原论文中用的是6个，这里针对一个编码器模块展开。每个编码器模块包含两个子模块：多头注意力机制；前馈神经网络。

  - Multi-head Attention

    多头注意力是使用多个普通注意力机制全连接得到的，因此完全可以使用矩阵进行并行计算。

    那么什么是注意力机制呢？简单而言就是使用Q，K，V分别作为查询，和键值对，因此会有对应的查询向量，键值对向量，也会在网络结构中有Q,K,V矩阵，用可学习的参数对q,k,v进行加权。

    缩放点击注意力的实现：

    ```python
    class ScaledDotProductAttention(nn.Module):
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()
    
        def forward(self, Q, K, V, attn_mask):  # 已经处理好的Q,K,V
            '''
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
            '''
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]  注意力评分函数
            scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
            
            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v] 得到加权后的V
            return context, attn
    ```

    结合下面的代码来看：

    ```python
    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super(MultiHeadAttention, self).__init__()
            self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
            self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
            self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        def forward(self, input_Q, input_K, input_V, attn_mask):
            '''
            input_Q: [batch_size, len_q, d_model]
            input_K: [batch_size, len_k, d_model]
            input_V: [batch_size, len_v(=len_k), d_model]
            attn_mask: [batch_size, seq_len, seq_len]
            '''
            residual, batch_size = input_Q, input_Q.size(0)
            # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
            Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
            K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
            V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
    
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
    
            # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
            context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
            context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
            output = self.fc(context) # [batch_size, len_q, d_model]
            return nn.LayerNorm(d_model).cuda()(output + residual), attn
    ```

    多头注意力机制的代码，这里可以用作自注意力，也可以用作编码器-解码器之间的注意力，d_model和embedding_size是一样的。因此对于自注意力输入的q,k,v都是```(batch_size, num_steps, embedding_dim)```维度，而且qkv是完全一样的！经过view和transpose方法调整数据维度后，Q,K,V都会变成```(batch_size, num_heads, num_steps, d_k)```,把这些数据送进缩放点击注意力，建议进行手推，可以得到数据维度```(batch_size, num_heads, len_q, len_v)```，接着对输出进行变换，经过一个全连接层把数据维度映射为```[batch_size, len_q, d_model]```，和输入是一样的，因此可以和输入做一个残差链接，之后再进行一次Layernorm。
    
    **在以上的介绍中，有两个问题没有进行说明：Mask，和Layernorm，下面逐一介绍。**
    
    Mask，为什么要使用，因为在注意力机制的运算中，需要使用到softmax函数，如果padding值是0的话，在softmax函数中会参与到运算，影响最终结果，因此需要想办法把所有的padding都给设置为一个极小值。
    
    那么在哪里设呢？最好的办法莫过于在softmax前。
    
    ```python
     def get_attn_pad_mask(seq_q, seq_k):
        '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
        '''
        batch_size, len_q = seq_q.size()  # len_q为序列长度，即有几个词元，len_k同样
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    
    attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
    ```
    
    ![](https://github.com/NICHOLASFATHER/dongxu_master_degree/blob/master/E81CFE61-3600-4119-82AC-AF18D1C9572E.png?raw=true)
    
    画了个图，大概是这样子，最后要在softmax之前填上极小值，即fill_mask
    
    Layernorm不在本文档介绍，可以参考仓库中专门介绍归一化的文档。
  
  - Feedforward Layer
  
    这个模块没什么可圈可点的，仅仅是两层全连接神经网络罢了。
  
    直接上代码，不解释：
  
    ```python
    class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
            super(PoswiseFeedForwardNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False)
            )
        def forward(self, inputs):
            '''
            inputs: [batch_size, seq_len, d_model]
            '''
            residual = inputs
            output = self.fc(inputs)
            return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
    ```
  
  单个编码器层的完整结构：
  
  ```python
  class EncoderLayer(nn.Module):
      def __init__(self):
          super(EncoderLayer, self).__init__()
          self.enc_self_attn = MultiHeadAttention()
          self.pos_ffn = PoswiseFeedForwardNet()
  
      def forward(self, enc_inputs, enc_self_attn_mask):
          '''
          enc_inputs: [batch_size, src_len, d_model]
          enc_self_attn_mask: [batch_size, src_len, src_len]
          '''
          # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
          enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
          enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
          return enc_outputs, attn
  ```
  
  这里输入的已经是embedding和位置编码后的数据，因此输入维度是```[batch_size, src_len, d_model]```，输出数据维度也不发生改变。

终极无敌完整编码器：

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
```

就是把上边所有的模块穿起来，对于最原始的数据出入维度是```[batch_size, num_steps]```(已经有了padding)，经过embedding层后变成```[batch_size, num_steps, embedding_size(same with d_model)]```， 然后加上位置编码，数据维度不变。然后把数据一次次得从到编码器核心层中，得到编码器最终的输出。

- decoder module

  解码器的一个模块和编码器的模块类似，只不过多了一个与编码器交互的注意力机制子模块。

  直接上代码：

  ```python
  class DecoderLayer(nn.Module):
      def __init__(self):
          super(DecoderLayer, self).__init__()
          self.dec_self_attn = MultiHeadAttention()
          self.dec_enc_attn = MultiHeadAttention()
          self.pos_ffn = PoswiseFeedForwardNet()
  
      def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
          '''
          dec_inputs: [batch_size, tgt_len, d_model]
          enc_outputs: [batch_size, src_len, d_model]
          dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
          dec_enc_attn_mask: [batch_size, tgt_len, src_len]
          '''
          # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
          dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
          # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
          dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
          dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
          return dec_outputs, dec_self_attn, dec_enc_attn
  ```

  可以看到创建了两个注意力机制模块，一个是自注意力，一个是和编码器之间的注意力。

  编码器的mask也非常有讲究，先介绍Sequence mask

  - Sequence mask

    Transformer是并行计算的，因此在解码器端的输入是并行的，然而序列生成总是要一个个得生成，所以这里需要一个mask来遮住那些未来的序列。

    ```python
    def get_attn_subsequence_mask(seq):
        '''
        seq: [batch_size, tgt_len]
        '''
        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
        subsequence_mask = torch.from_numpy(subsequence_mask).byte()
        return subsequence_mask # [batch_size, tgt_len, tgt_len]
    ```

    ```np.triu```用来创建上三角阵，而且主对角线的元素全为0。

    在之后的处理中，1代表padding，因此相当于每次多输入一个词元。

终极无敌解码器

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda() # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda() # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda() # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs) # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns
```

为什么自注意力的mask要相加？我画了个图解：

![](https://github.com/NICHOLASFATHER/dongxu_master_degree/blob/master/C6CB1E4F-DF5D-4136-9375-ED913FEA77A6.png?raw=true)

因为目标序列总有个结束的时候吧，而一个批量内不可能大家都是一块停，所以有些序列已经停了的时候，我们就把他遮住，防止无限的扩张。

**Transformer**

就是把之前的编码器和解码器给拼接起来

```python
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
```

说白了就是在Decoder输出后，把输出映射到目标词典的大小，之后再取其中概率最大的输出就好。

至此，模型部分介绍完毕。

## 数据处理

下面介绍预处理数据，直接上代码

```python
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
# 导入各种可能会用到的包
# S: Symbol that shows starting of decoding input
# E: Symbol that shows endding of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
# 这里为了方便理解手工创建了一个数据集
# Padding Should be Zero
# 手动给出整个词典的对应关系，以及词典大小
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}  # 即上边的字典反过来，方便后边生成的时候索引
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
      enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
      dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
      dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]

      enc_inputs.extend(enc_input)
      dec_inputs.extend(dec_input)
      dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
    super(MyDataSet, self).__init__()
    self.enc_inputs = enc_inputs
    self.dec_inputs = dec_inputs
    self.dec_outputs = dec_outputs
  
  def __len__(self):
    return self.enc_inputs.shape[0]  # 可以查询一共有多少条数据
  
  def __getitem__(self, idx):
    return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
```

## 模型训练

有了数据有了网络，自然是要开始训练啦。

```python
model = Transformer().cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

for epoch in range(30):  # 一共训练30个epoch
    for enc_inputs, dec_inputs, dec_outputs in loader: # 每个batch有两条数据
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()  # 放到gpu上加速训练
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs) 
      loss = criterion(outputs, dec_outputs.view(-1)) # 最后是个分类问题，最好是用crossEntopy损失函数
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

      optimizer.zero_grad()  # 清空梯度
      loss.backward()  # 反向传播
      optimizer.step()  # 梯度下降更新参数
```

## 模型测试

```python
def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)  # 输入源语言
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)  # 转换为相同的数据类型
    terminal = False  # 结束推理的标志
    next_symbol = start_symbol  # 以开始符号输入
    while not terminal:         
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)            
    return dec_input

# Test
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.cuda()
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
```

