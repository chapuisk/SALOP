import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.despine(left=True, right=True, bottom=True, top=True)
sns.set_style('white')

file = open("/Users/kevinchapuis/Development/Models/Gama/TsunamiUA/outputs/raw_outpus.csv", 'r')
df = pd.read_csv(file, sep=';', header=0)

headers = list(df.columns)[:-1]
par = headers[:4]
out = headers[4:]
o1 = out[0]
o2 = out[1]

ndf = pd.DataFrame(columns=['category', o1, o2])
i = 0
for vals in df.groupby(par)[out]:
    x = vals[1][o1].to_numpy()
    y = vals[1][o2].to_numpy()
    for j in range(len(x)):
        ndf.loc[len(ndf)] = {'category': "_"+str(i), o1: x[j]/vals[0][3], o2: y[j]*60}
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
ax.set(xlabel='Proportion of evacuated agents', ylabel='Average time to start evacuating (sec)')
sns.scatterplot(data=ndf, x=o1, y=o2, hue='category', ax=ax, palette=sns.color_palette("tab20"), s=10)
plt.show()
