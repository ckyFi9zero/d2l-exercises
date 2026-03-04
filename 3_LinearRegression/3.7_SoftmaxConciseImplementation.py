import torch
from torch import nn
import matplotlib
matplotlib.use('Agg') # 确保 Docker 无显示器环境正常
from d2l import torch as d2l
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from torchvision import transforms

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
print(f"{'Epoch':<10} | {'Loss':<10} | {'Train Acc':<10} | {'Test Acc':<10}")


epochs_list = []
train_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(num_epochs):
    # 训练一个周期
    # train_epoch_ch3 返回 (训练损失均值, 训练准确率均值)
    train_metrics = d2l.train_epoch_ch3(net, train_iter, loss, trainer)
    
    # 在测试集上评估精度
    test_acc = d2l.evaluate_accuracy(net, test_iter)
    
    # 1. 在终端打印结果 (解决你的需求)
    print(f"{epoch+1:<10} | {train_metrics[0]:<10.4f} | {train_metrics[1]:<10.4f} | {test_acc:<10.4f}")
    
    # 记录数据用于最后绘图
    epochs_list.append(epoch + 1)
    train_loss_list.append(train_metrics[0])
    train_acc_list.append(train_metrics[1])
    test_acc_list.append(test_acc)

# 训练结束后，手动创建图片（这样不会触发 display 打印）
plt.figure(figsize=(3.5, 2.5))
plt.plot(epochs_list, train_loss_list, '-', label='train loss')
plt.plot(epochs_list, train_acc_list, 'm--', label='train acc')
plt.plot(epochs_list, test_acc_list, 'g-.', label='test acc')
plt.xlabel('epoch')
plt.legend()
plt.grid()

plt.savefig('3.7_training_progress.png')
print("训练完成！结果已保存至 3.7_training_progress.png")

def predict_and_save(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.savefig('3.7_prediction_results.png')
    plt.close() # 显式关闭，防止内存占用和多余输出
    print("预测结果图已保存至: 3.7_prediction_results.png")

predict_and_save(net, test_iter)
