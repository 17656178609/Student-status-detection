import matplotlib.pyplot as plt

# 定义数据
conditions = ['YOLOv8x', 'YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l','yolov8l_MHSA_C2f_Dcn', 'YOLOv5_x', 'Faster-Rcnn','Ours']
mAP_05 = [0.721, 0.653, 0.7, 0.715, 0.726, 0.712, 0.704, 0.733, 0.763]


# 创建图形和坐标轴
plt.figure(figsize=(12, 5))

# 绘制mAP@0.5数据
plt.plot(conditions, mAP_05, marker='o', label='mAP@0.5')


# 添加标记
for i, mAP in enumerate(mAP_05):
    plt.text(conditions[i], mAP, f"{mAP:.3f}", ha='center', va='bottom')

plt.axhline(y=0.721, color='green', linestyle='dashed', label='Baseline|mAP@0.5')
plt.axhline(y=0.763, color='red', linestyle='dashed', label='Ours|mAP@0.5')




# 设置标题和标签

plt.ylim(0.5, 0.8)  # 设置y轴范围

# 添加图例
plt.legend()

# 自动调整布局以避免标签重叠
plt.tight_layout()

# 显示图形
plt.grid(True)
plt.show()
