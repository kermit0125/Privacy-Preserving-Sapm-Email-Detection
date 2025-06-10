import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载预处理后的数据集
X_train = pd.read_csv('./X_train.csv')
y_train = pd.read_csv('./y_train.csv')

# 分析函数
def analyze_preprocessed_data(X, y):
    # 文件保存路径
    output_dir = './analysis_results/'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 目标变量分布
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind='bar', color=['skyblue', 'orange'])
    plt.title('Target Variable Distribution')
    plt.xlabel('Class (0 = Non-Spam, 1 = Spam)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig(output_dir + 'target_distribution.png')
    plt.close()

    # 2. 部分特征的分布直方图
    plt.figure(figsize=(12, 8))
    X.iloc[:, :10].hist(bins=30, figsize=(12, 10), layout=(2, 5), color='skyblue', edgecolor='black')
    plt.suptitle('Distribution of First 10 Features', fontsize=16)
    plt.savefig(output_dir + 'features_distribution.png')
    plt.close()

    # 3. 偏度和峰度分析
    skewness = X.skew()
    kurtosis = X.kurtosis()
    skew_kurt_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
    skew_kurt_df.to_csv(output_dir + 'skewness_kurtosis.csv')

    # 4. 特征相关性热力图
    corr_matrix = X.corr().abs()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(output_dir + 'correlation_heatmap.png')
    plt.close()

    print("分析结果保存完毕，以下是生成文件：")
    print(f"1. 目标变量分布图：{output_dir}target_distribution.png")
    print(f"2. 特征分布直方图：{output_dir}features_distribution.png")
    print(f"3. 偏度和峰度结果表：{output_dir}skewness_kurtosis.csv")
    print(f"4. 特征相关性热力图：{output_dir}correlation_heatmap.png")

# 调用分析函数
analyze_preprocessed_data(X_train, y_train)
