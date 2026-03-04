import torch
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from d2l import torch as d2l

# 数据
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# 画 ReLU
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
plt.savefig('4.1_relu.png')
plt.close()

# 求梯度
y.backward(torch.ones_like(x), retain_graph=True)

# 画梯度
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
plt.savefig('4.1_relu_grad.png')
plt.close()

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
plt.savefig('4.1_sigmoid.png')
plt.close()



# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
plt.savefig('4.1_sigmoid_grad.png')
plt.close()

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
plt.savefig('4.1_tanh.png')
plt.close()

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
plt.savefig('4.1_tanh_grad.png')
plt.close()