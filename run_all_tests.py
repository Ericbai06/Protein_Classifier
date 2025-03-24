#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
蛋白质分类任务 - 所有测试脚本
此脚本可以运行主模型、正则化测试、学习率测试和特征工程测试
"""

import argparse
import time
import os

# 使用try-except处理导入可能出现的错误
try:
    from main_lr import main as main_test
except ImportError:
    print("警告: 无法导入main_lr模块，主模型测试将不可用")
    main_test = None

try:
    from regularization_test import test_regularization_impact
except ImportError:
    print("警告: 无法导入regularization_test模块，正则化测试将不可用")
    test_regularization_impact = None

try:
    from learning_rate_test import test_learning_rate_impact
except ImportError:
    print("警告: 无法导入learning_rate_test模块，学习率测试将不可用")
    test_learning_rate_impact = None

try:
    from feature_engineering_test import feature_engineering_test
except ImportError:
    print("警告: 无法导入feature_engineering_test模块，特征工程测试将不可用")
    feature_engineering_test = None

def run_all_tests(args):
    """
    运行所有测试并保存结果
    """
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 保存当前时间作为测试ID
    test_id = time.strftime("%Y%m%d_%H%M%S")
    print(f"开始测试，测试ID: {test_id}")
    
    # 1. 运行主模型测试
    if args.main:
        print("\n======== 运行主模型测试 ========")
        # 创建一个参数对象传递给main_test
        main_args = argparse.Namespace(ent=args.ent, verbose=True)
        if main_test:
            main_test(main_args)
    
    # 2. 运行正则化系数测试
    if args.regularization:
        print("\n======== 运行正则化系数测试 ========")
        if test_regularization_impact:
            c_values, train_accs, test_accs = test_regularization_impact()
            # 保存结果到文件
            with open(f'results/regularization_{test_id}.txt', 'w') as f:
                f.write("正则化系数测试结果\n")
                f.write("C值\t训练准确率\t测试准确率\n")
                for i, c in enumerate(c_values):
                    f.write(f"{c}\t{train_accs[i]:.4f}\t{test_accs[i]:.4f}\n")
    
    # 3. 运行学习率测试
    if args.learning_rate:
        print("\n======== 运行学习率测试 ========")
        if test_learning_rate_impact:
            alpha_values, train_accs, test_accs = test_learning_rate_impact()
            # 保存结果到文件
            with open(f'results/learning_rate_{test_id}.txt', 'w') as f:
                f.write("学习率测试结果\n")
                f.write("学习率\t训练准确率\t测试准确率\n")
                for i, alpha in enumerate(alpha_values):
                    f.write(f"{alpha}\t{train_accs[i]:.4f}\t{test_accs[i]:.4f}\n")
    
    # 4. 运行特征工程测试
    if args.feature_engineering:
        print("\n======== 运行特征工程测试 ========")
        if feature_engineering_test:
            results = feature_engineering_test()
            # 保存结果到文件
            with open(f'results/feature_engineering_{test_id}.txt', 'w') as f:
                f.write("特征工程测试结果\n")
                f.write("方法名称\t训练准确率\t测试准确率\n")
                for method_name, res in results.items():
                    f.write(f"{method_name}\t{res['avg_train_acc']:.4f}\t{res['avg_test_acc']:.4f}\n")
    
    print(f"\n所有测试完成，结果已保存到results目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="蛋白质分类任务 - 所有测试")
    parser.add_argument('--main', action='store_true', help="运行主模型测试")
    parser.add_argument('--regularization', action='store_true', help="运行正则化系数测试")
    parser.add_argument('--learning_rate', action='store_true', help="运行学习率测试")
    parser.add_argument('--feature_engineering', action='store_true', help="运行特征工程测试")
    parser.add_argument('--all', action='store_true', help="运行所有测试")
    parser.add_argument('--ent', action='store_true', help="使用fea.py中的feature_extraction()函数从文件加载数据")
    
    args = parser.parse_args()
    
    # 如果指定了--all，则运行所有测试
    if args.all:
        args.main = True
        args.regularization = True
        args.learning_rate = True
        args.feature_engineering = True
    
    # 如果没有指定任何测试，默认运行所有测试
    if not (args.main or args.regularization or args.learning_rate or args.feature_engineering):
        args.main = True
        args.regularization = True
        args.learning_rate = True
        args.feature_engineering = True
    
    run_all_tests(args) 