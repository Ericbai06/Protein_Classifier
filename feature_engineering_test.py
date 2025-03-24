import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from main_lr import LRModel, data_preprocess
from plot_utils import configure_plot_style

def feature_engineering_test():
    """
    测试不同特征工程方法对蛋白质分类任务的影响
    包括：
    1. 数据标准化方法比较
    2. 特征选择方法
    3. 主成分分析(PCA)降维
    """
    # 设置图表样式
    configure_plot_style()
    
    print("加载数据...")
    args = argparse.Namespace(ent=False)
    original_data_list, original_target_list = data_preprocess(args)
    
    # 为了节省时间，我们选择前5个数据集进行测试
    selected_datasets = range(5)
    data_list = [original_data_list[i] for i in selected_datasets]
    target_list = [original_target_list[i] for i in selected_datasets]
    
    # 不同的特征工程方法
    feature_eng_methods = {
        "原始数据": lambda x: x,
        "标准化(StandardScaler)": lambda x: StandardScaler().fit_transform(x),
        "归一化(MinMaxScaler)": lambda x: MinMaxScaler().fit_transform(x),
        "稳健缩放(RobustScaler)": lambda x: RobustScaler().fit_transform(x),
        "PCA降维(n_components=0.95)": lambda x: PCA(n_components=0.95).fit_transform(x),
        "特征选择(SelectKBest, k=10)": lambda x, y: SelectKBest(f_classif, k=min(10, x.shape[1])).fit_transform(x, y)
    }
    
    results = {}
    
    for method_name, method_func in feature_eng_methods.items():
        print(f"\n====== 特征工程方法: {method_name} ======")
        
        task_acc_train = []
        task_acc_test = []
        
        for i in range(len(data_list)):
            train_data, test_data = data_list[i]
            train_targets, test_targets = target_list[i]
            
            # 应用特征工程方法
            if method_name == "特征选择(SelectKBest, k=10)":
                train_data_transformed = method_func(train_data, train_targets)
                # 为测试数据使用相同的特征选择
                selector = SelectKBest(f_classif, k=min(10, train_data.shape[1]))
                selector.fit(train_data, train_targets)
                test_data_transformed = selector.transform(test_data)
            else:
                # 对训练数据应用转换
                train_data_transformed = method_func(train_data)
                # 对测试数据应用相同的转换
                test_data_transformed = method_func(test_data)
            
            # 创建模型
            model = LRModel(C=1.0, max_iter=1000)
            
            # 训练模型
            model.train(train_data_transformed, train_targets)
            
            # 评估模型
            train_accuracy = model.evaluate(train_data_transformed, train_targets)
            test_accuracy = model.evaluate(test_data_transformed, test_targets)
            
            print(f"数据集 {i+1}/{len(data_list)} - 训练准确率: {train_accuracy:.4f}, 测试准确率: {test_accuracy:.4f}")
            
            task_acc_train.append(train_accuracy)
            task_acc_test.append(test_accuracy)
        
        # 计算平均准确率
        avg_train_acc = sum(task_acc_train) / len(task_acc_train)
        avg_test_acc = sum(task_acc_test) / len(task_acc_test)
        
        print(f"{method_name} - 平均训练准确率: {avg_train_acc:.4f}")
        print(f"{method_name} - 平均测试准确率: {avg_test_acc:.4f}")
        
        results[method_name] = {
            'avg_train_acc': avg_train_acc,
            'avg_test_acc': avg_test_acc
        }
    
    # 结果比较与可视化
    print("\n====== 特征工程方法比较 ======")
    print(f"{'方法名称':<30} {'训练准确率':<15} {'测试准确率':<15}")
    print("-" * 60)
    
    method_names = []
    train_accuracies = []
    test_accuracies = []
    
    for method_name, res in results.items():
        print(f"{method_name:<30} {res['avg_train_acc']:<15.4f} {res['avg_test_acc']:<15.4f}")
        method_names.append(method_name)
        train_accuracies.append(res['avg_train_acc'])
        test_accuracies.append(res['avg_test_acc'])
    
    # 创建条形图比较不同方法
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 直接设置中文字体 - 备选方案
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    x = np.arange(len(method_names))
    width = 0.35
    
    ax.bar(x - width/2, train_accuracies, width, label='训练准确率')
    ax.bar(x + width/2, test_accuracies, width, label='测试准确率')
    
    ax.set_ylabel('准确率')
    ax.set_title('不同特征工程方法的性能比较')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_engineering_comparison.png')
    plt.show()
    
    return results

if __name__ == "__main__":
    print("开始测试不同特征工程方法...")
    feature_engineering_test() 