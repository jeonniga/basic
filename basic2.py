"""
    내용
    일반적인 딥러닝 책에서 시작 예제로 잡는 프로그램
    x값을 입력해 그 값 그대로 y값으로 출력하는 기능
    다항식 처리 예제.

    모델
    x1*100 + x2*200 + 150

    작성
    임학수 2020-01-04

    목적 
    강의용
"""

# Tensorflow v2.0 이후의 케라스 라이브러리
from tensorflow import keras 

# 수치처리 자료구조
import numpy as np 
from random import randint 

# 차트 그리기 위한 라이브러리
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

# 학습데이터 생성 
x = [ randint(0, 100) for i in range(100000) ] 
y = [ x[i*2]*100 + x[i*2+1]*200 + 150 for i in range(50000) ] 
x = np.asarray(x).astype(np.float32) 
x = x.reshape((50000, 2)) 
y = np.asarray(y).astype(np.float32) 
y = y.reshape((50000, 1)) 

# 학습 데이터 그래프 그리기 
xs = x[:,0] 
ys = x[:,1] 
zs = y 
fig = plt.figure(figsize=(12,12)) 
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(xs, ys, zs) 
ax.view_init(15, 15) 
plt.show()

# 검증 데이터 생성 
test_x = [ randint(0, 100) for i in range(100) ] 
test_y = [ test_x[i*2]*100 + test_x[i*2+1]*200 + 150 for i in range(50) ] 

test_x = np.asarray(test_x).astype(np.float32) 
test_x = test_x.reshape((50, 2)) 
test_y = np.asarray(test_y).astype(np.float32) 
test_y = test_y.reshape((50, 1)) 

# 모델 생성 
model = keras.Sequential() 
model.add(keras.layers.Dense(1, input_shape=(2,))) 
rmsprop = keras.optimizers.RMSprop(lr=0.02) 
model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=[keras.metrics.mse]) 
model.summary() 

# 학습 
model.fit(x, y, epochs=10) 

# 검증 데이터로 모델 평가 (verbose = 1 출력 모드) 
model.evaluate(test_x, test_y, verbose=1) 
ret = model.predict(test_x[:3]) 
print("예측 데이터 : ", ret) 
print("실제 데이터 : ", test_y[:3])
