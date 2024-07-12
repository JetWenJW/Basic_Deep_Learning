import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# 混淆矩陣(Confusion Matrix)
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1, 1 ]     # Actual Value
y_pred = [0, 1, 0, 1, 0, 1, 0, 1 ]     # Predict Value

# 混淆矩陣(Confusion Matrix)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f'TP={tp}, FP={fp}, TN={tn}, FN={fn}')  


fix, ax = plt.subplots(figsize = (2.5, 2.5))

# 1:藍色, 0:白色
ax.matshow([[1, 0], [0, 1]], cmap = plt.cm.Blues, alpha = 0.3)

ax.text(x = 0, y = 0, s = tp, va = 'center', ha = 'center')
ax.text(x = 1, y = 0, s = fp, va = 'center', ha = 'center')
ax.text(x = 0, y = 1, s = tn, va = 'center', ha = 'center')
ax.text(x = 1, y = 1, s = fn, va = 'center', ha = 'center')

plt.xlabel('Actual', fontsize = 20)
plt.xlabel('Predict', fontsize = 20)

plt.xticks([0, 1], ['T', 'F'])
plt.yticks([0, 1], ['P', 'N'])
plt.show()


# Accuracy
m = metrics.Accuracy()
m.update_state(y_true, y_pred)

print(f'Accuracy Rate: {m.result().numpy()}')
print(f'Validation = {(tp + tn) / (tp + tn + fp + fn)}')

# Precision
m = metrics.Precision()
m.update_state(y_true, y_pred)

print(f'Accuracy Rate: {m.result().numpy()}')
print(f'Validation = {(tp) / (tp + fp)}')

# Recall
m = metrics.Recall()
m.update_state(y_true, y_pred)

print(f'Recall Rate: {m.result().numpy()}')
print(f'Validation = {(tp) / (tp + fn)}')

# 依資料檔data/auc_data.csv計算AUC
# 讀取資料檔
import pandas as pd
df=pd.read_csv('./data/auc_data.csv')
df


from sklearn.metrics import roc_curve, roc_auc_score, auc

# fpr：假陽率，tpr：真陽率, threshold：各種決策門檻
fpr, tpr, threshold = roc_curve(df['actual'], df['predict'])
print(f'假陽率={fpr}\n\n真陽率={tpr}\n\n決策門檻={threshold}')


# 繪圖
auc1 = auc(fpr, tpr)
## Plot the result
plt.title('ROC/AUC')
plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()    

m = metrics.AUC()
m.update_state(df['actual'], df['predict'])

print(f'AUC:{m.result().numpy()}')













