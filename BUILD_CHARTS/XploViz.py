import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.despine(left=True, right=True, bottom=True, top=True)
sns.set()
#sns.set_style('white')

## READ DATA

file = open("/Users/kevinchapuis/Development/Models/Gama/TsunamiUA/outputs/ofat.csv", 'r')
df = pd.read_csv(file, sep=',', header=0)

headers = list(df.columns)[:-1]
idx_portable = 2
idx_scenario = 3
idx_sirene = 5
idx_data = 6

final_step = max(df["final step"])

ndf = df.groupby(headers[:5])
saves = dict()
decision = dict()

for runs in ndf:
    for run in runs[1].iterrows():
        params = run[1][:5].values
        params = tuple(params)
        if not params in saves:
            saves[params] = list()
        saves[params].append(run[1][5:97].values.tolist())
        if not params in decision:
            decision[params] = list()
        decision[params].append(run[1][97:].values.tolist())

outputDF = pd.DataFrame(columns=["prop portable", 'connaissance sirene', "scenario", 'tick', 'saves', 'decision'])

for runs in saves:
    thesaves = [float(sum(col))/len(col) for col in zip(*saves[runs])]
    thedecisions = [float(sum(col)) / len(col) for col in zip(*decision[runs])]
    for idx in range(final_step):
        outputDF = pd.concat([outputDF, pd.DataFrame({'prop portable': runs[1], 'connaissance sirene': runs[4],
                                                      "scenario": runs[2], 'tick': idx, 'saves': thesaves[idx],
                                                  'decision': thedecisions[idx]}, index=[0])])

## PLOT ALERT SCENARIO

xes = outputDF["prop portable"].unique()
yes = outputDF["connaissance sirene"].unique()

ses = ['both confirmed', 'both with strong confirmation']
ies = ['both & go at signal', 'siren & go at signal', 'siren confirmed']
bes = ['both & go at signal', 'broadcast & go at signal', 'broadcast confirmed']

mes = ['both confirmed', 'both with strong confirmation', 'broadcast confirmed']

yvar = xes[1:]
expvar = 'prop portable' #"connaissance sirene"
xvar = mes

huevar = 'connaissance sirene' #None
outvar = 'decision' #'saves'

fig = plt.figure()
#sns.lineplot(outputDF[(outputDF['scenario'] == 'siren confirmed')], x='tick', y=outvar, hue='connaissance sirene')
sns.lineplot(outputDF[(outputDF['scenario'] == 'broadcast confirmed')], x='tick', y=outvar, hue=expvar, legend="full")

# fig, axes = plt.subplots(len(yvar), len(xvar), sharey=True)
# fig.suptitle('phone proportion VS alert knowledge with decision styles')
# for idx, vp in enumerate(yvar):
#     for scn in xvar:
#         sns.lineplot(outputDF[(outputDF[expvar] == vp) & (outputDF['scenario'] == scn)],
#                      ax=axes[idx, xvar.index(scn)], x='tick', y=outvar, hue=huevar, legend=None)
#
# for i, ax in enumerate(axes):
#     for j, r in enumerate(ax):
#         if i == len(yvar)-1:
#             r.set_xlabel(xvar[j])
#         if j == 0:
#             r.set_ylabel(yvar[i])
#
# plt.tight_layout()
plt.show()
