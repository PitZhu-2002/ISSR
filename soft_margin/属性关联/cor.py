import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_excel('data.xls')
sns.set(rc={'figure.figsize':(20,10)})
sns.heatmap(data.iloc[:,:-1].corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="PiYG")
plt.show()

