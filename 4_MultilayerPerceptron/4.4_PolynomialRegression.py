import math
import numpy as np
import torch
from torch import nn
import matplotlib
# 确保在 Docker 等无界面环境下不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 1. 数据生成逻辑
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray 转换为 tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

def evaluate_loss(net, data_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 2. 修改训练函数以保存图片并增加终端输出
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400, filename='plot.png'):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    
    print(f"\n开始训练: {filename}")
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            train_l = evaluate_loss(net, train_iter, loss)
            test_l = evaluate_loss(net, test_iter, loss)
            animator.add(epoch + 1, (train_l, test_l))
    
    # 终端输出最终权重
    print(f'最终训练损失: {train_l:.6f}')
    print(f'最终测试损失: {test_l:.6f}')
    print('权重 (weight):', net[0].weight.data.numpy())
    
    # 保存图片
    plt.savefig(filename)
    print(f"图片已保存为: {filename}")
    plt.close() # 关闭当前画布，防止多图叠加

# 3. 三种实验对比

# 情况 A: 三阶多项式函数拟合 (正常拟合)
# 从多项式特征中选择前4个维度，即 1, x, x^2/2!, x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:], filename='4.4_poly_normal.png')

# 情况 B: 线性函数拟合 (欠拟合)
# 从多项式特征中选择前2个维度，即 1 和 x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:], filename='4.4_poly_underfitting.png')

# 情况 C: 高阶多项式函数拟合 (过拟合)
# 从多项式特征中选取所有维度 (20阶)
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500, filename='4.4_poly_overfitting.png')
