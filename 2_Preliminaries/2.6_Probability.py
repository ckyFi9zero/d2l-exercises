import numpy as np
import matplotlib
# 强制指定使用 Agg 后端，确保在 Docker/服务器环境下运行不弹出窗口报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. 进行 m=500 组试验，每组抽取 n=10 个样本。改变 m 和 n，观察实验结果。
def solve_problem_1(m=500, n=10):
    # 模拟：每组抽取 n 个服从标准正态分布的样本，计算其均值
    # 这是一个演示中心极限定理 (CLT) 的经典实验
    data = np.random.randn(m, n)
    sample_means = data.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(sample_means, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of Sample Means (m={m}, n={n})')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    output_file = f'2.6.1_m{m}_n{n}.png'
    plt.savefig(output_file)
    plt.close()
    print(f"统计实验完成，结果已保存至: {output_file}")


# 2. 给定 P(A) 和 P(B)，计算 P(A ∪ B) 和 P(A ∩ B) 的上限和下限。
#    （提示：使用维恩图来辅助思考）
def solve_problem_2(p_a, p_b):
    # 根据概率论基本性质（及维恩图推导）：
    # 1. P(A ∩ B) 的下限：max(0, P(A) + P(B) - 1)
    # 2. P(A ∩ B) 的上限：min(P(A), P(B))
    intersection_lower = max(0, p_a + p_b - 1)
    intersection_upper = min(p_a, p_b)
    
    # 3. P(A ∪ B) 的下限：max(P(A), P(B))
    # 4. P(A ∪ B) 的上限：min(1, P(A) + P(B))
    union_lower = max(p_a, p_b)
    union_upper = min(1, p_a + p_b)
    
    print(f"P(A ∩ B) 的范围: [{intersection_lower:.2f}, {intersection_upper:.2f}]")
    print(f"P(A ∪ B) 的范围: [{union_lower:.2f}, {union_upper:.2f}]")


# 3. 假设我们有一系列随机变量 A、B 和 C，其中 B 仅依赖于 A，而 C 仅依赖于 B。
#    你能简化联合概率 P(A, B, C) 吗？（提示：这是一个马尔可夫链的例子）

# 根据条件概率的链式法则：P(A, B, C) = P(A) * P(B | A) * P(C | A, B)
# 由于题目给定 C 仅依赖于 B（满足马尔可夫性质），则 P(C | A, B) = P(C | B)
# 简化后的结果为：P(A, B, C) = P(A) * P(B | A) * P(C | B)


# 4. 在 2.6 节中，第一场测试更准确。为什么不运行第一场测试两次，
#    而是同时运行第一场和第二场测试？
# 解答：
# 1. 独立性与多样性：如果第一场测试的两次运行是完全相关的（例如输入相同且算法确定），
#    那么第二次运行不会提供任何新信息。
# 2. 误差抵消：不同的测试（第一场和第二场）通常具有不同的偏置或错误模式。
#    同时运行两个不同的测试可以提供多维度的反馈，有助于降低系统性误差，
#    这在集成学习（Ensemble Learning）中被称为“多样性增益”。


if __name__ == "__main__":
    # 运行第一题（可以尝试修改参数）
    solve_problem_1(m=500, n=10)
    
    # 运行第二题（示例：P(A)=0.6, P(B)=0.5）
    solve_problem_2(0.6, 0.5)
