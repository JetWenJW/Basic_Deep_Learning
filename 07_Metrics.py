import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# 混淆矩陣(Confusion Matrix)所需的函數
from sklearn.metrics import confusion_matrix

# 定義實際值和預測值
y_true = [0, 0, 0, 1, 1, 1, 1, 1]     # 真實標籤
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]     # 預測標籤

# 計算混淆矩陣
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()   # 計算並展平混淆矩陣
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')                # 打印 TP, FP, TN, FN 的值

# 繪製混淆矩陣的可視化
fix, ax = plt.subplots(figsize=(2.5, 2.5))                  # 創建一個 2.5x2.5 的圖形和子圖

# 1:藍色, 0:白色
ax.matshow([[1, 0], [0, 1]], cmap=plt.cm.Blues, alpha=0.3)  # 使用藍色顯示混淆矩陣的格子

# 在矩陣上添加文本標籤
ax.text(x=0, y=0, s=tp, va='center', ha='center')  # 在 (0, 0) 添加 TP 的值
ax.text(x=1, y=0, s=fp, va='center', ha='center')  # 在 (1, 0) 添加 FP 的值
ax.text(x=0, y=1, s=tn, va='center', ha='center')  # 在 (0, 1) 添加 TN 的值
ax.text(x=1, y=1, s=fn, va='center', ha='center')  # 在 (1, 1) 添加 FN 的值

plt.xlabel('Actual', fontsize=20)   # 設置 x 軸標籤
plt.ylabel('Predict', fontsize=20)  # 設置 y 軸標籤

plt.xticks([0, 1], ['T', 'F'])      # 設置 x 軸刻度
plt.yticks([0, 1], ['P', 'N'])      # 設置 y 軸刻度
plt.show()                          # 顯示圖形

# 計算準確率 (Accuracy)
m = metrics.Accuracy()              # 創建 Accuracy 計算實例
m.update_state(y_true, y_pred)      # 更新準確率計算狀態

print(f'Accuracy Rate: {m.result().numpy()}')               # 打印準確率
print(f'Validation = {(tp + tn) / (tp + tn + fp + fn)}')    # 使用公式計算準確率

# 計算精確率 (Precision)
m = metrics.Precision()         # 創建 Precision 計算實例
m.update_state(y_true, y_pred)  # 更新精確率計算狀態

print(f'Precision Rate: {m.result().numpy()}')  # 打印精確率
print(f'Validation = {(tp) / (tp + fp)}')       # 使用公式計算精確率

# 計算召回率 (Recall)
m = metrics.Recall()                # 創建 Recall 計算實例
m.update_state(y_true, y_pred)      # 更新召回率計算狀態

print(f'Recall Rate: {m.result().numpy()}')     # 打印召回率
print(f'Validation = {(tp) / (tp + fn)}')       # 使用公式計算召回率

# 依資料檔 data/auc_data.csv 計算 AUC
# 讀取資料檔
import pandas as pd
df = pd.read_csv('./data/auc_data.csv')  # 讀取 CSV 文件到 DataFrame
print(df)                                # 顯示 DataFrame 內容

from sklearn.metrics import roc_curve, roc_auc_score, auc

# fpr：假陽率，tpr：真陽率, threshold：各種決策門檻
fpr, tpr, threshold = roc_curve(df['actual'], df['predict'])            # 計算 ROC 曲線的假陽率、真陽率和決策門檻
print(f'假陽率 = {fpr}\n\n真陽率 = {tpr}\n\n決策門檻 = {threshold}')    # 打印假陽率、真陽率和決策門檻

# 繪製 ROC 曲線
auc1 = auc(fpr, tpr)  # 計算 AUC（ROC 曲線下的面積）
## Plot the result
plt.title('ROC/AUC')                                            # 設置圖形標題
plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)  # 繪製 ROC 曲線
plt.legend(loc='lower right')                                   # 顯示圖例，位置設為右下角
plt.plot([0, 1], [0, 1], 'r--')                                 # 繪製對角線（隨機猜測的線）
plt.xlim([0, 1])                                                # 設置 x 軸範圍
plt.ylim([0, 1])                                                # 設置 y 軸範圍
plt.ylabel('True Positive Rate')                                # 設置 y 軸標籤
plt.xlabel('False Positive Rate')                               # 設置 x 軸標籤
plt.show()                                                      # 顯示圖形

m = metrics.AUC()                               # 創建 AUC 計算實例
m.update_state(df['actual'], df['predict'])     # 更新 AUC 計算狀態

print(f'AUC: {m.result().numpy()}')             # 打印 AUC 值
