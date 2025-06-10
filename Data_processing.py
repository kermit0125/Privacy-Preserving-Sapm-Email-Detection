# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# 加载数据集
file_path = './emails.c sv'
data = pd.read_csv(file_path)

# 数据预处理步骤

# 1. 移除第一列 (Email No.)，因为它是标识符，与模型无关
data_cleaned = data.drop(columns=['Email No.'])

# 2. 提取特征 (X) 和目标变量 (y)
X = data_cleaned.iloc[:, :-1]  # 所有特征列（除最后一列）
y = data_cleaned.iloc[:, -1]   # 最后一列是标签

# 3. 特征归一化到 [0, 1] 范围
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 4. 划分训练集和测试集（80:20比例）
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

# 5. 保存预处理结果到本地以供后续使用
X_train.to_csv('./X_train.csv', index=False)
X_test.to_csv('./X_test.csv', index=False)
y_train.to_csv('./y_train.csv', index=False)
y_test.to_csv('./y_test.csv', index=False)


print("数据集预处理完成，文件已保存：")
print("训练特征集：./X_train.csv")
print("测试特征集：./X_test.csv")
print("训练标签集：./y_train.csv")
print("测试标签集：./y_test.csv")
