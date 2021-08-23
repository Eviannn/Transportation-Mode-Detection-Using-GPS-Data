############################################################################################
# 作者：戴晚锐
# 项目：毕业设计随机森林模型选取特征（代码6）
############################################################################################

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_excel('D:\\final year project code\\Feature(after cleaning).xlsx')

features = ['Max Speed(m/s)', '95% Speed(m/s)', '75% Speed(m/s)', 'Mean Speed(m/s)', 'Non 0 Mean Speed(m/s)', 'Speed Std']

# features = ['Max Acceleration(m/s^2)', '95% Acceleration(m/s^2)', '75% Acceleration(m/s^2)',
#            'Mean Acceleration(m/s^2)', 'Non 0 Mean Acceleration(m/s^2)', 'Acceleration Std']

X = data[features]
y = data['Labels']

names = features
rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', oob_score=True, max_depth=4, random_state=233)
rf.fit(X, y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True))
