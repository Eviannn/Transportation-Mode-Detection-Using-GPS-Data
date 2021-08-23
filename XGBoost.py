############################################################################################
# 作者：戴晚锐
# 项目：毕业设计XGBoost模型（代码8）
############################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel('D:\\final year project code\\Feature.xlsx')
features = ['Max Speed(m/s)', 'Non 0 Mean Speed(m/s)', 'Speed Std', 'Max Acceleration(m/s^2)',
            'Acceleration Std', 'Non 0 Mean Acceleration(m/s^2)']

# 划分数据集
X = data[features]
y = data['Labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=233)

# X_test.to_excel('data_test.xls')

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义分层K折交叉验证
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=233)


# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=(0.5, 1.01), cv=skf, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='f1_weighted')
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='f1_weighted')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("XGBoost模型：训练集Macro weighted F1-score: %.2f%%" % (train_scores_mean[-1] * 100.0))
    print("XGBoost模型交叉验证Macro weighted F1-score: %.2f%%" % (test_scores_mean[-1] * 100.0))
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


clf = XGBClassifier(
    objective='multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
    #silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    booster='gbtree',
    #nthread=4,# cpu 线程数 默认最大
    learning_rate=0.1, # 如同学习率
    min_child_weight=1,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=3, # 构建树的深度，越大越容易过拟合
    gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=0.7, # 随机采样训练样本 训练实例的子采样比
    #max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=0.7, # 生成树时进行的列采样
    reg_lambda=0.8,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #reg_alpha=0, # L1 正则项参数
    #scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    num_class=5, # 类别数，多分类与 multisoftmax 并用
    n_estimators=100, #树的个数
    seed=233, #随机种子
    eval_metric='mlogloss'
)

plot_learning_curve(clf, u"XGBoost learning curve", X_train, y_train)
plt.show()


eval_set = [(X_test, y_test)]
clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# 对测试集进行预测
y_pred = clf.predict(X_test)

# 计算测试集得分
test_accuracy = precision_score(y_test, y_pred, average='weighted')

print("XGBoost模型：测试集Macro weighted F1-score: %.2f%%" % (test_accuracy*100.0))

# 输出预测结果
output = pd.DataFrame({'Labels': y_test, 'Labels_pred': y_pred})
output.to_csv('predictions.csv', index=False)

'''
# 显示重要特征
plot_importance(clf)
plt.show()

output = pd.DataFrame({'Labels': y_test, 'Labels_pred': y_pred})
output.to_csv('prediction.csv', index=False)
'''