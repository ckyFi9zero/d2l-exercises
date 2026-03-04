import torch
from torch import nn
import matplotlib
# 确保在 Docker 等无界面环境下不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from d2l import torch as d2l

# 1. 定义 Dropout 层
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)

# 2. 定义模型结构
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模式下才使用 dropout
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

# 3. 自定义训练函数，用于 Docker 终端输出和保存图片
def train_and_save(net, train_iter, test_iter, loss, num_epochs, trainer, filename):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    
    print(f"\n开始训练模式: {filename}")
    print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 50)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 训练损失、训练准确率、样本数
        net.train()
        for X, y in train_iter:
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        
        train_metrics = (metric[0] / metric[2], metric[1] / metric[2])
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
        # 终端输出
        print(f"{epoch+1:<8} | {train_metrics[0]:<10.4f} | {train_metrics[1]:<10.4f} | {test_acc:<10.4f}")

    plt.savefig(filename)
    print(f"训练完成，图表已保存为: {filename}")
    plt.close() # 必须关闭，否则两张图会叠加

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net_scratch = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
trainer_scratch = torch.optim.SGD(net_scratch.parameters(), lr=lr)
train_and_save(net_scratch, train_iter, test_iter, loss, num_epochs, trainer_scratch, '4.6_dropout_scratch.png')

net_seq = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(dropout2),
    nn.Linear(256, 10)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net_seq.apply(init_weights)
trainer_seq = torch.optim.SGD(net_seq.parameters(), lr=lr)
train_and_save(net_seq, train_iter, test_iter, loss, num_epochs, trainer_seq, '4.6_dropout_sequential.png')


def predict_and_save(net, test_iter, filename):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:6].reshape((6, 28, 28)), 1, 6, titles=titles[0:6])
    plt.savefig(filename)
    print(f"预测图片已保存为: {filename}")

predict_and_save(net_seq, test_iter, '4.6_dropout_predict.png')
