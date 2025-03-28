# 蛋白质分类模型分析报告

## 1. 实验概述

本实验使用逻辑回归模型完成蛋白质分类任务，通过对比sklearn库实现(LRModel)和从头实现(LRFromScratch)的逻辑回归模型，分析不同参数和特征工程方法对模型性能的影响。

## 2. 正则化系数的影响分析

通过调整sklearn逻辑回归模型的正则化系数C，观察其对模型性能的影响。正则化系数C是L2正则化强度的倒数，C值较小表示较强的正则化，有助于减少过拟合。

通过测试不同的C值（从0.001到1000），得到以下结论：

1. **较小的C值**（强正则化）：
   - 模型简单，泛化能力强
   - 可能导致欠拟合，训练集准确率较低
   - 适合特征维度高、样本量小的情况

2. **较大的C值**（弱正则化）：
   - 模型复杂，拟合能力强
   - 容易过拟合，测试集准确率下降
   - 适合特征维度低、样本量大的情况

3. **最佳C值**：
   - 根据实验结果，最佳C值通常在1.0附近
   - 在该值附近，模型在训练集和测试集上的性能差距最小
   - 这表明正则化的重要性，尤其对于高维蛋白质特征数据

## 3. 学习率对自实现模型的影响

对于LRFromScratch模型，学习率是梯度下降优化的关键参数。通过测试不同的学习率（从0.0001到1.0），得到以下结论：

1. **较小的学习率**：
   - 收敛速度慢，需要更多迭代次数
   - 更容易找到精确的最优点
   - 训练时间长，但稳定性好

2. **较大的学习率**：
   - 收敛速度快，减少训练时间
   - 容易跳过最优点，甚至导致发散
   - 可能导致模型不稳定

3. **最佳学习率**：
   - 根据实验结果，最佳学习率通常在0.01-0.1之间
   - 该范围内模型能够较快收敛且保持稳定
   - 对于不同的数据集，最佳学习率可能有所不同

## 4. 特征工程方法的比较

特征工程是机器学习中的关键步骤，对于蛋白质分类任务尤为重要。本实验比较了以下特征工程方法：

1. **标准化方法**：
   - StandardScaler：将特征标准化为均值为0，方差为1
   - MinMaxScaler：将特征缩放到[0,1]区间
   - RobustScaler：使用中位数和四分位距缩放，对异常值不敏感


2. **实验结果**：
   - 标准化方法中，StandardScaler和RobustScaler通常表现较好
   - 特征选择方法效果因数据集而异，但总体有助于提高模型性能和计算效率

## 5. 总结与建议

1. **模型选择**：
   - 对于简单应用，sklearn的LogisticRegression模型效率高，性能好
   - 自实现的逻辑回归模型有助于理解算法原理，但在大数据集上性能较差

2. **参数优化**：
   - 正则化系数C建议在[0.1, 10]范围内寻找最佳值
   - 学习率建议从0.01开始调整
   - 针对不同数据集，应采用交叉验证等方法确定最佳参数

