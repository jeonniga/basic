"""
    내용
    일반적인 딥러닝 책에서 시작 예제로 잡는 프로그램
    x값을 입력해 그 값 그대로 y값으로 출력하는 기능
    x,y를 학습해 y값을 예측하는 linear egression 예제임.

    모델
    aX + b = Y
    2X + 1 = Y

    작성
    임학수 2020-01-04

    목적 
    강의용
"""

# 수치 자료구조 
import numpy as np

# 케라스 딥러링 라이브러리
from keras.models import Sequential
from keras.layers import Dense


# 입력값 X,  특성 1개
x = np.array([1,2,3,4,5,6,7,8,9,10])
# 출력값 Y, 입력값 X가 주어 졌을때 출력값 Y
y = np.array([3,5,7,9,11,13,15,17,19,21])


# 순차모델 생성 (모델을 순차적으로 구성하겠다)
model = Sequential()
# 레이어 추가 : 출력차원 1개, 입력차원 1개(특징 1개), 활성함수 linear
model.add( Dense(1, input_dim=1, activation='linear') )

# 모델 컴파일, 정밀도를 매트릭으로 하여 판정 
# 최적화 알고리즘 : adam, 
# 손실함수 : MSE (평균제곱오차)
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# 실제학습을 배치사이즈 1, 반복학습 횟수 500으로 설정
model.fit(x, y, epochs=500, batch_size=1, verbose=1, shuffle=True)


# 모델 평가
loss, acc = model.evaluate(x, y, batch_size=1)
# 로스값과 정밀도 출력
print('loss: ', loss)
print('acc: ', acc)


# 예측을 위한 x값 
x_in = [1,3]
# x값을 이용해 예측
ypred = model.predict(x=x_in)

# 예측된 값을 입력값과 함께 출력
for i, val in enumerate(x_in):
    print(i, ':', val, '(', val*2+1, ')->', ypred[i])