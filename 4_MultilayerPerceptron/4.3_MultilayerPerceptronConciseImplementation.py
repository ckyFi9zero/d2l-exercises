import torch
from torch import nn
import matplotlib
# 确保在 Docker 等无界面环境下不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 1. 定义模型
# 第一题可以在这里修改隐藏层神经元数量（256）
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), 
                    nn.ReLU(),
                    nn.Linear(256, 10))

# 2. 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

# 3. 设置超参数和数据
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 4. 自定义训练函数（用于终端输出和保存图表）
def train_ch3_docker(net, train_iter, test_iter, loss, num_epochs, trainer):
    # 初始化 Animator 用于内部数据收集
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 50)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
        net.train()
        
        for X, y in train_iter:
            # 计算梯度并更新参数
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            
            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        
        # 计算每轮的指标
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        
        # 更新绘图数据
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
        # 终端实时输出
        print(f"{epoch+1:<8} | {train_metrics[0]:<10.4f} | {train_metrics[1]:<10.4f} | {test_acc:<10.4f}")

    # 保存训练曲线图
    plt.savefig('4.3_train_metrics_sequential.png')
    print("-" * 50)
    print("训练完成！图表已保存为 '4.3_train_metrics_sequential.png'")

# 5. 执行训练
train_ch3_docker(net, train_iter, test_iter, loss, num_epochs, trainer)

# 6. 预测并保存结果图
def predict_ch3_docker(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.savefig('4.3_predict_results_sequential.png')
    print("预测图片已保存为 '4.3_predict_results_sequential.png'")

predict_ch3_docker(net, test_iter)
