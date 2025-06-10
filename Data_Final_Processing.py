# 导入必要的库
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# 加载训练集和测试集
X_train = pd.read_csv('./X_train.csv')
y_train = pd.read_csv('./y_train.csv')
X_test = pd.read_csv('./X_test.csv')
y_test = pd.read_csv('./y_test.csv')

# 1. 上采样 - 使用 SMOTE 平衡类别分布
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 确保 y_resampled 是 Series 类型
if isinstance(y_resampled, pd.DataFrame):
    y_resampled = y_resampled.squeeze()

# 检查类别分布是否平衡
print("类别分布（上采样后）：")
print(y_resampled.value_counts())

# 2. 主成分分析 (PCA)
# 设置主成分数目，假设我们希望保留 95% 的累计方差
pca = PCA(n_components=0.95, random_state=42)
X_train_final = pca.fit_transform(X_resampled)

# 输出 PCA 的结果
print(f"PCA 降维后训练集的形状：{X_train_final.shape}")
print(f"累计解释方差比：{sum(pca.explained_variance_ratio_):.2f}")

# 测试集直接应用相同的 PCA 变换
X_test_final = pca.transform(X_test)

# 保存结果到本地
pd.DataFrame(X_train_final).to_csv('./X_train_final.csv', index=False)
pd.DataFrame(X_test_final).to_csv('./X_test_final.csv', index=False)
y_resampled.to_csv('./y_train_final.csv', index=False)
y_test.to_csv('./y_test_final.csv', index=False)

print("上采样和 PCA 完成，结果保存到文件：")
print("训练集特征：./X_train_final.csv")
print("测试集特征：./X_test_final.csv")
print("训练集标签：./y_train_final.csv")
print("测试集标签：./y_test_final.csv")
