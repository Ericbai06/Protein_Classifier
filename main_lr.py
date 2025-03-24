import os
import argparse
import numpy as np
import pandas as pd 

from sklearn.preprocessing import label_binarize, StandardScaler
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser


class LRModel:
    """
    Logistic Regression模型类，使用sklearn的LogisticRegression实现。
    
    这个类是对sklearn LogisticRegression的封装，提供了初始化、训练和评估方法。
    """
    
    def __init__(self, C=1.0, max_iter=1000, solver='liblinear', random_state=42):
        """
        初始化Logistic Regression模型。
        
        参数:
        - C (float): 正则化强度的倒数，较小的值表示更强的正则化
        - max_iter (int): 求解器收敛的最大迭代次数
        - solver (str): 优化问题的算法，可选: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
        - random_state (int): 随机数生成器的种子
        """
        self.lr = LogisticRegression(
            penalty='l2',
            dual=False,
            tol=0.0001,
            C=C,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            verbose=0,
            warm_start=False,
            n_jobs=-1,
            l1_ratio=None
        )

    def train(self, train_data, train_targets):
        """
        训练Logistic Regression模型。
        
        参数:
        - train_data (array-like): 训练数据
        - train_targets (array-like): 训练数据对应的目标值
        """
        self.lr.fit(train_data, train_targets)
        
    def predict(self, data):
        """
        预测样本的类别。
        
        参数:
        - data (array-like): 需要预测的样本
        
        返回:
        - array-like: 预测的类别标签
        """
        return self.lr.predict(data)
        
    def evaluate(self, data, targets):
        """
        评估模型在给定数据上的性能。
        
        参数:
        - data (array-like): 需要评估的数据
        - targets (array-like): 数据对应的真实目标值
        
        返回:
        - float: 模型在给定数据上的准确率
        """
        return self.lr.score(data, targets)


class LRFromScratch:
    """
    从头实现的逻辑回归模型。
    
    使用梯度下降算法优化参数。
    """

    def __init__(self, alpha=0.01, epochs=1000, tol=1e-4):
        """
        初始化Logistic Regression模型。
        
        参数:
        - alpha (float): 学习率
        - epochs (int): 最大迭代次数
        - tol (float): 收敛容差
        """
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        
    def sigmoid(self, z):
        """
        Sigmoid激活函数。
        
        参数:
        - z (array-like): 输入值
        
        返回:
        - array-like: Sigmoid值
        """
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y, w):
        """
        计算逻辑回归的损失函数。
        
        参数:
        - X (array-like): 特征矩阵
        - y (array-like): 目标值
        - w (array-like): 权重向量
        
        返回:
        - float: 损失值
        """
        m = len(y)
        z = np.dot(X, w)
        h = self.sigmoid(z)
        # 计算交叉熵损失
        epsilon = 1e-5  # 避免log(0)
        cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost
        
    def gradient_descent(self, X, y, w, alpha, num_iters):
        """
        执行梯度下降优化。
        
        参数:
        - X (array-like): 特征矩阵
        - y (array-like): 目标值
        - w (array-like): 初始权重向量
        - alpha (float): 学习率
        - num_iters (int): 最大迭代次数
        
        返回:
        - array-like: 优化后的权重向量
        - list: 损失历史
        """
        m = len(y)
        cost_history = []
        
        for i in range(num_iters):
            z = np.dot(X, w)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            
            # 更新权重
            w_prev = w.copy()
            w = w - alpha * gradient
            
            # 计算损失
            cost = self.compute_cost(X, y, w)
            cost_history.append(cost)
            
            # 检查收敛性
            if i > 0 and abs(cost_history[i] - cost_history[i-1]) < self.tol:
                break
                
            # 每100轮打印一次损失值
            if (i+1) % 100 == 0:
                print(f"迭代 {i+1}: 损失 = {cost}")
                
        return w, cost_history
        
    def train(self, train_data, train_targets):
        """
        训练逻辑回归模型。
        
        参数:
        - train_data (array-like): 训练数据
        - train_targets (array-like): 训练数据对应的目标值
        """
        # 添加偏置项
        X = np.c_[np.ones((train_data.shape[0], 1)), train_data]
        y = train_targets
        
        # 初始化权重
        w = np.zeros(X.shape[1])
        
        # 使用梯度下降算法优化权重
        self.w, self.cost_history = self.gradient_descent(X, y, w, self.alpha, self.epochs)
    
    def predict(self, X):
        """
        预测样本的类别。
        
        参数:
        - X (array-like): 样本
        
        返回:
        - array-like: 预测的类别标签
        """
        # 添加偏置项
        X = np.c_[np.ones((X.shape[0], 1)), X]
        z = np.dot(X, self.w)
        h = self.sigmoid(z)
        return np.where(h >= 0.5, 1, 0)
    
    def evaluate(self, data, targets):
        """
        评估逻辑回归模型的性能。
        
        参数:
        - data (array-like): 需要评估的数据
        - targets (array-like): 数据对应的真实目标值
        
        返回:
        - float: 模型在给定数据上的准确率
        """
        predictions = self.predict(data)
        accuracy = np.mean(predictions == targets)
        return accuracy


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # 55 classification tasks
        task_col = cast.iloc[:, task]
        
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []
        
        # Process each protein sample
        for i in range(len(task_col)):
            # Value 1: positive train sample
            if task_col.iloc[i] == 1:
                train_data.append(diagrams[i])
                train_targets.append(1)
            # Value 2: negative train sample
            elif task_col.iloc[i] == 2:
                train_data.append(diagrams[i])
                train_targets.append(0)
            # Value 3: positive test sample
            elif task_col.iloc[i] == 3:
                test_data.append(diagrams[i])
                test_targets.append(1)
            # Value 4: negative test sample
            elif task_col.iloc[i] == 4:
                test_data.append(diagrams[i])
                test_targets.append(0)
        
        # Convert to numpy arrays for sklearn compatibility
        train_data = np.array(train_data)
        train_targets = np.array(train_targets)
        test_data = np.array(test_data)
        test_targets = np.array(test_targets)
        
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main(args):
    """
    主函数：加载数据、训练模型并评估性能。
    
    参数:
    - args: 包含命令行参数的命名空间对象
    """
    data_list, target_list = data_preprocess(args)

    # 模型比较 - 测试不同的模型和参数
    models = {
        'LR_sklearn': LRModel(C=1.0, max_iter=1000),  # 增加迭代次数
        'LR_scratch': LRFromScratch(alpha=0.01, epochs=1000, tol=1e-4),
    }

    results = {}
    
    # 运行所有选定的模型
    for model_name, model in models.items():
        print(f"\n====== 运行模型: {model_name} ======")
        
        task_acc_train = []
        task_acc_test = []
        
        for i in range(len(data_list)):
            train_data, test_data = data_list[i]
            train_targets, test_targets = target_list[i]

            # 数据标准化（仅对sklearn模型进行）
            if model_name.startswith('LR_sklearn'):
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)

            if hasattr(args, 'verbose') and args.verbose:
                print(f"处理数据集 {i+1}/{len(data_list)}")

            # 训练模型
            model.train(train_data, train_targets)

            # 评估模型
            train_accuracy = model.evaluate(train_data, train_targets)
            test_accuracy = model.evaluate(test_data, test_targets)

            if hasattr(args, 'verbose') and args.verbose:
                print(f"数据集 {i+1}/{len(data_list)} - 训练准确率: {train_accuracy:.4f}, 测试准确率: {test_accuracy:.4f}")

            task_acc_train.append(train_accuracy)
            task_acc_test.append(test_accuracy)

        # 计算平均准确率
        avg_train_acc = sum(task_acc_train) / len(task_acc_train)
        avg_test_acc = sum(task_acc_test) / len(task_acc_test)
        
        print(f"{model_name} - 平均训练准确率: {avg_train_acc:.4f}")
        print(f"{model_name} - 平均测试准确率: {avg_test_acc:.4f}")
        
        results[model_name] = {
            'avg_train_acc': avg_train_acc,
            'avg_test_acc': avg_test_acc
        }
    
    # 比较不同模型的性能
    print("\n====== 模型性能比较 ======")
    print(f"{'模型名称':<20} {'训练准确率':<15} {'测试准确率':<15}")
    print("-" * 50)
    for model_name, res in results.items():
        print(f"{model_name:<20} {res['avg_train_acc']:<15.4f} {res['avg_test_acc']:<15.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument('--ent', action='store_true', help="使用fea.py中的feature_extraction()函数从文件加载数据")
    parser.add_argument('--verbose', action='store_true', help="显示详细输出信息")
    args = parser.parse_args()
    main(args)

