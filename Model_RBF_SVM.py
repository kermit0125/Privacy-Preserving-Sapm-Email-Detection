import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 创建结果文件夹
output_dir = './svm_model_results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载处理后的训练集和测试集
X_train = pd.read_csv('./X_train_final.csv')
X_test = pd.read_csv('./X_test_final.csv')
y_train = pd.read_csv('./y_train_final.csv')
y_test = pd.read_csv('./y_test_final.csv')

# 确保 y_train 和 y_test 为 Series 类型
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.squeeze()
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.squeeze()

# 1. 构建 SVM 模型
# 使用 RBF 核的支持向量机
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)

# 2. 模型预测
y_pred = model.predict(X_test)

# 3. 模型评估
# 分类报告
classification_report_str = classification_report(y_test, y_pred)
with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
    f.write(classification_report_str)

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
plt.title('Confusion Matrix (SVM)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# 4. 保存模型
joblib.dump(model, os.path.join(output_dir, 'svm_model.pkl'))

print(f"SVM 模型已训练完成，所有结果保存到文件夹：{output_dir}")
