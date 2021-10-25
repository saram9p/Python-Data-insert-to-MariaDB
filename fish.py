import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import sqlalchemy as db

pd_bream_length = pd.read_csv("./fish/bream_length.csv")
pd_bream_weight = pd.read_csv("./fish/bream_weight.csv")
pd_smelt_length = pd.read_csv("./fish/smelt_length.csv")
pd_smelt_weight = pd.read_csv("./fish/smelt_length.csv")

np_bream_length = pd_bream_length.to_numpy()
np_bream_weight = pd_bream_weight.to_numpy()
np_smelt_length = pd_smelt_length.to_numpy()
np_smelt_weight = pd_smelt_weight.to_numpy()

bream_length = np_bream_length.flatten()
bream_weight = np_bream_weight.flatten()
smelt_length = np_smelt_length.flatten()
smelt_weight = np_smelt_weight.flatten()

print(bream_length.shape)
print(smelt_length.shape)

# 데이터 시각화
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

fish_length = np.concatenate((bream_length, smelt_length))
fish_weight = np.concatenate((bream_weight, smelt_weight))

print(fish_length.shape)
print(fish_weight.shape)

# fish_length, fish_weight 합치기

fish_data = np.column_stack((fish_length, fish_weight))

print(fish_data.shape)  # shape 확인
print(fish_data)

# 타겟 데이터 만들기

fish_target = np.concatenate((np.ones(34), np.zeros(13)))

print(fish_target)

# 데이터 셔플

index = np.arange(47)
np.random.shuffle(index)
print(index)

print(fish_data.shape)
print(fish_target.shape)

# 훈련데이터 80%
train_input = fish_data[index[:37]]
train_target = fish_target[index[:37]]

# 테스트 데이터 20%
test_input = fish_data[index[37:]]
test_target = fish_target[index[37:]]

# 데이터 시각화
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
