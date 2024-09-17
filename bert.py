import pandas as pd
import numpy as np

# 设置随机种子以便重现
np.random.seed(42)

# 生成测试数据
num_samples = 1000  # 生成 1000 个样本
length_of_label = 217  # 标签数组的长度

# 示例评论列表
sample_comments = [
    "这个产品很好用！",
    "我不太喜欢这个颜色。",
    "服务态度非常好。",
    "质量一般，价格偏高。",
    "非常满意，值得购买！",
    "不会再来了，太失望了。",
    "性价比高，推荐给朋友。",
    "包装很好，物流也很快。",
    "使用了一段时间，效果不错。",
    "不推荐，完全不符合我的期待。"
]

# 随机生成评论
comments = np.random.choice(sample_comments, num_samples)

# 生成随机标签（每个标签为长度为 217 的数组，值为 0 或 1）
labels = [np.random.randint(0, 2, length_of_label).tolist() for _ in range(num_samples)]

# 创建 DataFrame
data = pd.DataFrame({
    'text': comments,
    'label': labels
})

# 保存为 CSV 文件
data.to_csv('test_data.csv', index=False)

print("测试数据已生成并保存为 'test_data.csv'")
