import matplotlib.pyplot as plt

# 数据
years = [2020, 2021, 2022, 2023, 2024]
market_sizes_usd = [0, 0, 0, 830.8, 1316.2]  # 如果正常没有2020-2022年的数据，我们使用0以表示空值

# 绘制折线图
plt.plot(years, market_sizes_usd, marker='o')
plt.title('AI \u8f6f\u4ef6\u5e02\u573a\u89c4\u6a21\uff08\u5355\u4f4d\uff1a\u5341\u4ebf\u7f8e\u5143\uff09')
plt.xlabel('\u5e74\u4efd')
plt.ylabel('\u5e02\u573a\u89c4\u6a21')
plt.grid(True)
plt.show()
