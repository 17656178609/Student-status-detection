import matplotlib.pyplot as plt

# 定义数据
conditions = ['raw_YOLOv8', 'C2f_Res2block', 'EMA', 'MHSA', 'C2f_Res2block+EMA', 'C2f_Res2block+EMA+MHSA']
mAP_05 = [0.721, 0.753, 0.745, 0.752, 0.758, 0.763]
mAP_095 = [0.551, 0.584, 0.558, 0.567, 0.577, 0.586]

# 创建图形和坐标轴
plt.figure(figsize=(12, 5))

# 绘制mAP@0.5数据
plt.plot(conditions, mAP_05, marker='o', label='mAP@0.5')

# 绘制mAP@0.5:0.95数据
plt.plot(conditions, mAP_095, marker='o', label='mAP@0.5:0.95')

# 添加标记
for i, (mAP, mAP95) in enumerate(zip(mAP_05, mAP_095)):
    plt.text(conditions[i], mAP, f"{mAP:.3f}", ha='center', va='bottom')
    plt.text(conditions[i], mAP95, f"{mAP95:.3f}", ha='center', va='bottom')

plt.axhline(y=0.721, color='green', linestyle='dashed', label='Baseline|mAP@0.5')
plt.axhline(y=0.763, color='red', linestyle='dashed', label='C2f_Res2block+EMA+MHSA|mAP@0.5')

plt.axhline(y=0.551, color='green', linestyle='dashed', label='Baseline|mAP@0.5:0.95')
plt.axhline(y=0.586, color='red', linestyle='dashed', label='C2f_Res2block+EMA+MHSA|mAP@0.5:0.95')


# 设置标题和标签

plt.ylim(0.5, 0.8)  # 设置y轴范围

# 添加图例
plt.legend()

# 自动调整布局以避免标签重叠
plt.tight_layout()

# 显示图形
plt.grid(True)
plt.show()
