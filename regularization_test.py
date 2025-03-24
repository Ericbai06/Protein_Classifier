import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import label_binarize, StandardScaler
from main_lr import LRModel, data_preprocess
from plot_utils import configure_plot_style

def test_regularization_impact():
    """
    测试不同正则化系数C对LogisticRegression模型性能的影响
    """
    # 设置图表样式
    configure_plot_style()
    
    print("加载数据...")
    args = argparse.Namespace(ent=False)
    data_list, target_list = data_preprocess(args)
    
    # 测试不同的正则化系数
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    train_accs = []
    test_accs = []
    
    for c in c_values:
        print(f"\n测试正则化系数 C={c}")
        model = LRModel(C=c, max_iter=1000)  # 增加迭代次数到1000
        
        task_acc_train = []
        task_acc_test = []
        
        for i in range(len(data_list)):
            train_data, test_data = data_list[i]
            train_targets, test_targets = target_list[i]
            
            # 数据标准化
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)
            
            # 训练模型
            model.train(train_data, train_targets)
            
            # 评估模型
            train_accuracy = model.evaluate(train_data, train_targets)
            test_accuracy = model.evaluate(test_data, test_targets)
            
            task_acc_train.append(train_accuracy)
            task_acc_test.append(test_accuracy)
        
        # 计算平均准确率
        avg_train_acc = sum(task_acc_train) / len(task_acc_train)
        avg_test_acc = sum(task_acc_test) / len(task_acc_test)
        
        print(f"C={c} - 平均训练准确率: {avg_train_acc:.4f}")
        print(f"C={c} - 平均测试准确率: {avg_test_acc:.4f}")
        
        train_accs.append(avg_train_acc)
        test_accs.append(avg_test_acc)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    # 直接设置中文字体 - 备选方案
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
        
    plt.plot(c_values, train_accs, 'bo-', label='训练准确率')
    plt.plot(c_values, test_accs, 'ro-', label='测试准确率')
    plt.xscale('log')
    plt.xlabel('正则化系数 C')
    plt.ylabel('准确率')
    plt.title('不同正则化系数C对模型性能的影响')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 确保所有元素都能正确显示
    plt.savefig('regularization_impact.png')
    plt.show()
    
    # 打印最佳结果
    best_c_index = np.argmax(test_accs)
    best_c = c_values[best_c_index]
    best_test_acc = test_accs[best_c_index]
    print(f"\n最佳正则化系数 C={best_c} - 测试准确率: {best_test_acc:.4f}")
    
    return c_values, train_accs, test_accs

if __name__ == "__main__":
    print("开始测试不同正则化系数的影响...")
    test_regularization_impact() 