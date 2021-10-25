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

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
