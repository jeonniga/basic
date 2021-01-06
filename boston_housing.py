"""
    내용
    보스턴 지역의 집값을 예측하는 프로그램으로 regression의 대표적인 예제.

    작성
    임학수 2020-01-04

    목적 
    강의용
"""

# 데이터 가공 모듈
import numpy as np
from pandas import DataFrame

# 데이터 가시화 모듈
import matplotlib.pyplot as plt

# 기계학습 모듈
from sklearn import linear_model
from sklearn.datasets import load_boston
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## 데이터 준비
# 보스턴 데이터 셋 임포트
boston = load_boston()

# 설명변수들을 DataFrame으로 변환
df = DataFrame(boston.data, columns = boston.feature_names)

# 목적변수를 DataFrame에 추가
df['MEDV'] = np.array(boston.target)

# 최초 5행을 표시
df.head()

# 오브젝트 생성
model = linear_model.Ridge()

# fit함수에서 파라미터 추정
model.fit(boston.data,boston.target)

# 회귀계수를 출력
print(model.coef_)
print(model.intercept_)


# 75%를 학습용, 25%를 검증용 데이터로 하기 위해 분할
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size = 0.25, random_state = 100)

# 학습용 데이터에서 파라미터 추정
model.fit(X_train, y_train)

# 작성한 모델로부터 예측(학습용, 검증용 모델 사용)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# 학습용, 검증용 각각에서 잔차를 플롯
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'gray', marker = 'o', label = 'Train Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'blue', marker = 's', label = 'Test Data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# 범례를 왼쪽 위에 표시
plt.legend(loc = 'upper left')

# y = 0의 직선을 그림
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'black')
plt.xlim([0, 50])
plt.show()

# 학습용, 검증용 데이터에 대하여 평균제곱오차를 출력
print('MSE Train : %.3f, Test : %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

# 학습용, 검증용 데이터에 대하여 R^2를 출력
print('R^2 Train : %.3f, Test : %.3f' % (model.score(X_train, y_train), model.score(X_test, y_test)))
