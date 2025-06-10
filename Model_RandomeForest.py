import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 创建结果文件夹
output_dir = './RandomeForest_model_results/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载训练集和测试集
X_train = pd.read_csv('./X_train_final.csv')
X_test = pd.read_csv('./X_test_final.csv')
y_train = pd.read_csv('./y_train_final.csv')
y_test = pd.read_csv('./y_test_final.csv')

# 加载原始特征名称
original_feature_names = pd.read_csv('./X_train.csv').columns.tolist()

# 确保 y_train 和 y_test 为 Series 类型
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.squeeze()
if isinstance(y_test, pd.DataFrame):
    y_test = y_test.squeeze()

# 1. 构建随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
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
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# 4. 特征重要性分析
# 使用原始单词作为特征名称
feature_importances = pd.Series(model.feature_importances_, index=original_feature_names[:X_train.shape[1]])

# 可视化前10个重要特征
plt.figure(figsize=(12, 6))
feature_importances.nlargest(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Feature Importances')
plt.ylabel('Importance Score')
plt.xlabel('Feature Name')
plt.xticks(rotation=45, ha='right')  # 旋转标签，便于阅读

# 添加双引号到横坐标标签
plt.gca().set_xticklabels([f'"{feature}"' for feature in feature_importances.nlargest(10).index])

plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
plt.close()

# 保存模型
joblib.dump(model, os.path.join(output_dir, 'random_forest_model.pkl'))

print(f"随机森林模型已训练完成，所有结果保存到文件夹：{output_dir}")
