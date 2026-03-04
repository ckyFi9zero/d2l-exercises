import torch
from torch import nn
import matplotlib
# 强制使用 Agg 后端，确保在无显示器的 Docker 环境中不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 1. 数据准备
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2. 初始化参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 3. 定义模型
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 隐藏层
    return (H @ W2 + b2)   # 输出层

loss = nn.CrossEntropyLoss(reduction='none')

# 4. 辅助函数（用于评估和累加）
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 5. 自定义训练函数（实现终端输出和保存图表）
def train_ch3_docker(net, train_iter, test_iter, loss, num_epochs, updater):
    # 初始化 Animator 用于绘图（d2l 内部会收集数据）
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 50)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3) # 训练损失总和、训练准确度总和、样本数
        
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_acc = evaluate_accuracy(net, test_iter)
        
        # 将数据添加到动画器中
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
        # 终端实时输出
        print(f"{epoch+1:<8} | {train_metrics[0]:<10.4f} | {train_metrics[1]:<10.4f} | {test_acc:<10.4f}")

    # 保存训练曲线图
    plt.savefig('4.2_train_metrics.png')
    print("-" * 50)
    print("训练完成！图表已保存为 '4.2_train_metrics.png'")

# 6. 开始训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

train_ch3_docker(net, train_iter, test_iter, loss, num_epochs, updater)

# 7. 预测并保存结果图
def predict_ch3_docker(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    
    # 使用 d2l 的绘图函数
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    
    # 保存预测图片
    plt.savefig('4.2_predict_results.png')
    print("预测图片已保存为 '4.2_predict_results.png'")

predict_ch3_docker(net, test_iter)
