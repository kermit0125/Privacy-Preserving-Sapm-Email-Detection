# 🛡️ Privacy-Preserving Spam Email Detection / 隐私保护型垃圾邮件识别系统

This repository contains a machine learning pipeline for detecting spam emails while ensuring **user privacy**. The project uses classic ML models on word-frequency vectors extracted from emails and applies **dimensionality reduction** and **model optimization** to build a lightweight, privacy-preserving classification system.

本项目旨在构建一个兼顾隐私保护的垃圾邮件识别系统，利用邮件词频特征在本地提取关键词，通过降维和模型优化实现高效、轻量级的分类模型。

---

## 📁 Dataset / 数据集

- **Source**: [Kaggle - Email Spam Classification Dataset](https://www.kaggle.com/datasets)  
- **Format**: 5000+ labeled emails, using top 3000 word frequencies  
- **Privacy**: Only word occurrence counts are used (no content transmitted)

---

## ⚙️ Pipeline Overview / 流程概览

1. **Data Preprocessing**  
   - Remove identifiers  
   - Normalize features  
   - Split into training/testing sets  
   - Store processed CSVs  

2. **Dimensionality Reduction**  
   - SMOTE to balance classes  
   - PCA to reduce 3000D → ~95% variance retained  

3. **Modeling**  
   - Random Forest  
   - RBF-SVM  
   - Optimized and non-optimized versions  

4. **Evaluation & Visualization**  
   - Accuracy, Confusion Matrix, Feature Importance  
   - Heatmaps, Histograms, PCA result plots

---

## 🧪 Models Used / 模型算法

| Model         | Optimized | Accuracy | 特点 |
|---------------|-----------|----------|------|
| Random Forest | ✅         | ~91%     | 稳定、可解释性强 |
| RBF SVM       | ✅         | ~89%     | 精度高，对边界敏感 |
| Lightweight Model | ✅     | ~85%     | 用于边缘设备部署 |

所有模型均对比降维与非降维场景下的表现，并保存混淆矩阵与特征重要性图表。

---

## 📊 Results / 项目结果

- 🔒 No raw email content used throughout the pipeline  
- 🎯 Accuracy > 90% with Random Forest after PCA  
- 🧩 PCA retained ~95% variance with >95% feature reduction  
- 📉 Top spam indicator words identified (e.g., "free", "win", "offer")

所有可视化结果存储于：  
`/RandomeForest_model_results/`  
`/svm_model_results/`  
`/analysis_results/`

---

## 🧠 Project Focus / 项目亮点

- Emphasizes **edge processing** to avoid uploading full email text  
- Trains and evaluates models with a focus on **privacy-respecting representation**  
- Builds groundwork for **on-device spam filtering systems**  
- Includes full **EDA and feature correlation** visualizations

---

## 📂 File Structure / 主要文件结构

```bash
.
├── Data_processing.py                 # 数据清洗、归一化、划分
├── Data_Final_Processing.py          # SMOTE + PCA 降维
├── Model_RandomeForest.py            # 降维后 Random Forest
├── Model_RandomeForest_without_optimize.py # 原始数据 Random Forest
├── Model_RBF_SVM.py                  # 降维后 RBF SVM
├── Model_RBM_SVM_without_optimize.py # 原始数据 RBF SVM
├── Data_analysis.py                  # 分析目标变量与特征分布
├── README.md                         # 项目说明文件
