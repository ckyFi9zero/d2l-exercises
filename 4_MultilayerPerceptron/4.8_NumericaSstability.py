import torch
import matplotlib
# 确保在 Docker 等无界面环境下不报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

# d2l.plot 会调用 matplotlib 绘图
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

# 保存图片
plt.savefig('4.8_sigmoid_gradient.png')
print("绘图完成，图片已保存为: '4.8_sigmoid_gradient.png'")

# --- 第二部分：演示梯度爆炸/消失 ---
print("\n" + "="*30)
M = torch.normal(0, 1, size=(4,4))
print('初始矩阵 M:\n', M)

# 连续乘以 100 个随机矩阵
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('\n乘以 100 个随机矩阵后的 M:')
print(M)
print("="*30)

# 提示：观察输出，你会发现数值通常会变得极大（inf）或极小，这就是梯度爆炸或消失的直观体现。
