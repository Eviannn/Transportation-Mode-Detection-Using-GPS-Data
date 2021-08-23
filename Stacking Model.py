############################################################################################
# 作者：戴晚锐
# 项目：毕业设计Stacking模型融合（代码9）
############################################################################################

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from xgboost.sklearn import XGBClassifier as XGB
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
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

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 第一层模型
clfs = [
    XGB(n_estimators=100, learning_rate=0.1, min_child_weight=1, max_depth=3, gamma=0.1, subsample=0.7,
        reg_lambda=0.8, colsample_bytree=0.7, eval_metric='mlogloss', random_state=233),
    RF(n_estimators=100, max_depth=5, class_weight='balanced', random_state=233),
    ET(n_estimators=200, max_depth=8, class_weight='balanced', random_state=233),
    SVC(kernel='rbf', gamma=0.5, C=10, class_weight='balanced', probability=True, random_state=233)
]
# Stacking：训练集Macro weighted F1-score: 92.38%
# Stacking交叉验证Macro weighted F1-score: 89.47%

lr = LogisticRegression(multi_class="multinomial", solver="lbfgs")

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
    print("Stacking：训练集Macro weighted F1-score: %.2f%%" % (train_scores_mean[-1] * 100.0))
    print("Stacking交叉验证Macro weighted F1-score: %.2f%%" % (test_scores_mean[-1] * 100.0))
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


sclf = StackingClassifier(classifiers=clfs,
                          use_probas=True,
                          average_probas=True,
                          meta_classifier=lr)


plot_learning_curve(sclf, u"Stacking learning curve", X_train, y_train)
plt.show()

# 对测试集进行预测
sclf.fit(X_train, y_train)
y_pred = sclf.predict(X_test)

# 计算测试集得分
test_accuracy = precision_score(y_test, y_pred, average='weighted')
print("Stacking：测试集Macro weighted F1-score: %.2f%%" % (test_accuracy * 100.0))

# 输出预测结果
output = pd.DataFrame({'Labels': y_test, 'Labels_pred': y_pred})
output.to_csv('prediction.csv', index=False)

