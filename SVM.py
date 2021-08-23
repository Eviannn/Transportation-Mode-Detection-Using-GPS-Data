############################################################################################
# 作者：戴晚锐
# 项目：毕业设计SVM模型（代码7）
############################################################################################

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

data = pd.read_excel('D:\\final year project code\\Feature.xlsx')
features = ['Max Speed(m/s)', 'Non 0 Mean Speed(m/s)', 'Speed Std', 'Max Acceleration(m/s^2)',
            'Acceleration Std', 'Non 0 Mean Acceleration(m/s^2)']

# 划分数据集
X = data[features]
y = data['Labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=233)

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
    print("SVM模型：训练集Macro weighted F1-score: %.2f%%" % (train_scores_mean[-1] * 100.0))
    print("SVM模型交叉验证Macro weighted F1-score: %.2f%%" % (test_scores_mean[-1] * 100.0))
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


clf = SVC(kernel='rbf', gamma=0.5, C=10, class_weight='balanced', random_state=233)
plot_learning_curve(clf, u"SVM learning curve", X_train, y_train)
plt.show()

clf.fit(X_train, y_train)
# 对测试集进行预测
y_pred = clf.predict(X_test)

# 计算测试集得分
test_accuracy = precision_score(y_test, y_pred, average='weighted')

print("SVM模型：测试集Macro weighted F1-score: %.2f%%" % (test_accuracy*100.0))
