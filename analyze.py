# %%
'Hello world'
# %%
# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Read data here
df = pd.read_csv('data/term-deposit-marketing-2020.csv')

#%%
# Surf on data
print(df.info())
print('-------------------------------------------------')
print(df.describe())
print('-------------------------------------------------')
print(df.describe(include=['O']))
print('-------------------------------------------------')
print(df.columns.values)

# %%
# Analyze single correlation 
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'age', bins=20).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'balance', bins=10).fig.show()
#%%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'day', bins=10).fig.show()
#%%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'month', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'duration', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'loan', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'default', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'job', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'campaign', bins=10).fig.show()
#%%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'contact', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'housing', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'marital', bins=10).fig.show()
# %%
grid = sns.FacetGrid(df, col='y')
grid.map(plt.hist, 'education', bins=10).fig.show()


# %%
# Show yes or no counts
df['y'].value_counts()


# %%
# Show categorical correlation
# grid = sns.FacetGrid(df, row='age', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'duration', 'day', 'month', palette='deep').fig.show()
# grid.add_legend()


# %%
# Please you need to calculate all correlation between all features and column "y". 
# And continue the Show categorical/numerical correlation.