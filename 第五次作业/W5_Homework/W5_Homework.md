# 计算机视觉第五次作业

<div align = "center"> 岳东旭        2201212864       指导教师：张健</div>

## 1.手动推导

![IMG_0742](./assets/IMG_0742.PNG)

## 2.搭建两层全连接神经网络

```python
%matplotlib inline
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())       
```


```python
x.shape
```


    torch.Size([100, 1])


```python
y.shape
```


    torch.Size([100, 1])


```python
plt.scatter(x.numpy(), y.numpy())
```


    <matplotlib.collections.PathCollection at 0x1ce251747c0>


![png](output_3_1.png)
​    


### 搭建两层含有bias的全连接网络，隐藏层输出个数为20，激活函数都用sigmoid()


```python
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # self.net = nn.Sequential()
        self.linear_1 = nn.Linear(n_feature, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.sigmoid(self.linear_1(x))
        x = self.linear_2(x)
        return x
```


```python
net = Net(n_feature=1, n_hidden=20, n_output=1)     # define the network
print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(2000):
    prediction = net(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 20 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 't = %d, Loss=%.4f' % (t, loss.data.numpy()), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        plt.show()

plt.ioff()
# plt.show()
```

    Net(
      (linear_1): Linear(in_features=1, out_features=20, bias=True)
      (linear_2): Linear(in_features=20, out_features=1, bias=True)
    )


    D:\anaconda\lib\site-packages\torch\nn\functional.py:1960: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")




![png](output_6_2.png)
    




![png](output_6_3.png)
    




![png](output_6_4.png)
    




![png](output_6_5.png)
    




![png](output_6_6.png)
    




![png](output_6_7.png)
    




![png](output_6_8.png)
    




![png](output_6_9.png)
    




![png](output_6_10.png)
    




![png](output_6_11.png)
    




![png](output_6_12.png)
    




![png](output_6_13.png)
    




![png](output_6_14.png)
    




![png](output_6_15.png)
    




![png](output_6_16.png)
    




![png](output_6_17.png)
    




![png](output_6_18.png)
    




![png](output_6_19.png)
    




![png](output_6_20.png)
    




![png](output_6_21.png)
    




![png](output_6_22.png)
    




![png](output_6_23.png)
    




![png](output_6_24.png)
    




![png](output_6_25.png)
    




![png](output_6_26.png)
    




![png](output_6_27.png)
    




![png](output_6_28.png)
    




![png](output_6_29.png)
    




![png](output_6_30.png)
    




![png](output_6_31.png)
    




![png](output_6_32.png)
    




![png](output_6_33.png)
    




![png](output_6_34.png)
    




![png](output_6_35.png)
    




![png](output_6_36.png)
    




![png](output_6_37.png)
    




![png](output_6_38.png)
    




![png](output_6_39.png)
    




![png](output_6_40.png)
    




![png](output_6_41.png)
    




![png](output_6_42.png)
    




![png](output_6_43.png)
    




![png](output_6_44.png)
    




![png](output_6_45.png)
    




![png](output_6_46.png)
    




![png](output_6_47.png)
    




![png](output_6_48.png)
    




![png](output_6_49.png)
    




![png](output_6_50.png)
    




![png](output_6_51.png)
    




![png](output_6_52.png)
    




![png](output_6_53.png)
    




![png](output_6_54.png)
    




![png](output_6_55.png)
    




![png](output_6_56.png)
    




![png](output_6_57.png)
    




![png](output_6_58.png)
    




![png](output_6_59.png)
    




![png](output_6_60.png)
    




![png](output_6_61.png)
    




![png](output_6_62.png)
    




![png](output_6_63.png)
    




![png](output_6_64.png)
    




![png](output_6_65.png)
    




![png](output_6_66.png)
    




![png](output_6_67.png)
    




![png](output_6_68.png)
    




![png](output_6_69.png)
    




![png](output_6_70.png)
    




![png](output_6_71.png)
    




![png](output_6_72.png)
    




![png](output_6_73.png)
    




![png](output_6_74.png)
    




![png](output_6_75.png)
    




![png](output_6_76.png)
    




![png](output_6_77.png)
    




![png](output_6_78.png)
    




![png](output_6_79.png)
    




![png](output_6_80.png)
    




![png](output_6_81.png)
    




![png](output_6_82.png)
    




![png](output_6_83.png)
    




![png](output_6_84.png)
    




![png](output_6_85.png)
    




![png](output_6_86.png)
    




![png](output_6_87.png)
    




![png](output_6_88.png)
    




![png](output_6_89.png)
    




![png](output_6_90.png)
    




![png](output_6_91.png)
    




![png](output_6_92.png)
    




![png](output_6_93.png)
    




![png](output_6_94.png)
    




![png](output_6_95.png)
    




![png](output_6_96.png)
    




![png](output_6_97.png)
    




![png](output_6_98.png)
    




![png](output_6_99.png)
    




![png](output_6_100.png)
    




![png](output_6_101.png)
    





    <matplotlib.pyplot._IoffContext at 0x1ce1d35fb50>

