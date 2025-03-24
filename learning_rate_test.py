import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from main_lr import LRFromScratch, data_preprocess
from plot_utils import configure_plot_style

def test_learning_rate_impact():
    """
    测试不同学习率(alpha)对LRFromScratch模型性能的影响
    """
    # 设置图表样式
    configure_plot_style()
    
    print("加载数据...")
    args = argparse.Namespace(ent=False)
    data_list, target_list = data_preprocess(args)
    
    # 测试不同的学习率
    alpha_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    train_accs = []
    test_accs = []
    
    # 选择前5个数据集进行测试，以节省时间
    selected_datasets = range(5)
    selected_data = [data_list[i] for i in selected_datasets]
    selected_targets = [target_list[i] for i in selected_datasets]
    
    for alpha in alpha_values:
        print(f"\n测试学习率 alpha={alpha}")
        model = LRFromScratch(alpha=alpha, epochs=1000, tol=1e-4)
        
        task_acc_train = []
        task_acc_test = []
        
        for i in range(len(selected_data)):
            train_data, test_data = selected_data[i]
            train_targets, test_targets = selected_targets[i]
            
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
        
        print(f"alpha={alpha} - 平均训练准确率: {avg_train_acc:.4f}")
        print(f"alpha={alpha} - 平均测试准确率: {avg_test_acc:.4f}")
        
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
        
    plt.plot(alpha_values, train_accs, 'bo-', label='训练准确率')
    plt.plot(alpha_values, test_accs, 'ro-', label='测试准确率')
    plt.xscale('log')
    plt.xlabel('学习率 alpha')
    plt.ylabel('准确率')
    plt.title('不同学习率对LRFromScratch模型性能的影响')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # 确保所有元素都能正确显示
    plt.savefig('learning_rate_impact.png')
    plt.show()
    
    # 打印最佳结果
    best_alpha_index = np.argmax(test_accs)
    best_alpha = alpha_values[best_alpha_index]
    best_test_acc = test_accs[best_alpha_index]
    print(f"\n最佳学习率 alpha={best_alpha} - 测试准确率: {best_test_acc:.4f}")
    
    return alpha_values, train_accs, test_accs

if __name__ == "__main__":
    print("开始测试不同学习率的影响...")
    test_learning_rate_impact() 