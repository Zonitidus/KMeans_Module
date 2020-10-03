import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


data1 = pd.read_fwf('a1.txt', header = None)

normalized_df = ((data1-data1.mean())/data1.std())

plt.scatter(normalized_df[0].values, normalized_df[1].values)
plt.show()