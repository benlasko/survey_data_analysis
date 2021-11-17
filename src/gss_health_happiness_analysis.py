import pyreadstat

import pandas as pd
import numpy as np

import scipy.stats as stats
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns


def dataframe_overview(df):
    '''
    Prints the following for a quick overview of a dataframe:
        Information returned from .info()
        First five rows (.head())
        Shape (.shape)
        List of column names
        Information returned from .describe(). The "top" is the most common value. The "freq" is the most common value’s frequency.
        Total duplicate rows
        
    Parameter
    ----------
    df:  pd.DataFrame
        A Pandas DataFrame
        
    Returns
    ----------
       None.
    '''
    columns = df.columns.values.tolist()

    print("\u0332".join("INFO "))
    print(f'{df.info()}\n\n')
    print("\u0332".join("HEAD "))
    print(f'{df.head()}\n\n')
    print("\u0332".join("SHAPE "))
    print(f'{df.shape}\n\n')
    print("\u0332".join("COLUMNS "))
    print(f'{columns}\n\n')
    print("\u0332".join("COLUMN STATS "))
    print(f'{df.describe()}\n\n')
    print('\u0332'.join("TOTAL DUPLICATE ROWS "))
    print(f' {df.duplicated().sum()}')

    
def correlations(df, cols, round_dec=2):
    '''
    Prints correlation matrix and heatmap for chosen columns of dataframe.
    
    Parameters
    ----------
    df: Pandas dataframe
    
    cols: list  
        List of dataframe columns to find correlations for.  
    
    corr_round: int, default=2
        Number of decimals to round correlation values to.
    
    Returns
    ----------
    Correlation Matrix: dataframe
        Matrix of correlations.

    Heatmap:  seaborn heatmap
        Heatmap showing correlations for each row, annotated with correlation values in percentages.
    '''

    df1 = df.copy()
    df1 = df1[cols]
    corrs = df1.corr().round(round_dec)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(), cmap='coolwarm', robust=True, annot=True, fmt='.0%')
    plt.show()
    return corrs


def bootstrap(x, iterations=10000):
    lst = []
    for i in range(iterations):
        bootstrap = np.random.choice(x, size=len(x), replace=True)
        lst.append(bootstrap)
    return lst


def bootstrap_confidence_interval(sample, stat_function=np.mean, iterations=10000, ci=95):
    '''
    Bootstraps sample data.
    
    Parameters:
    sample: Numpy array
        1-d numeric data.
    
    stat_function: function, default=np.mean
        Function for calculating a sample statistic on data.
    
    iterations: int, default=10000
        Number of bootstrap samples to create.
    
    ci: int, default=95
        Percent of distribution encompassed by CI, 0<ci<100.
    '''
    
    bootstrap_samples = bootstrap(sample, iterations=iterations)
    bootstrap_samples_stat = list(map(stat_function, bootstrap_samples))
    low_bound = (100. - ci) / 2
    high_bound = 100. - low_bound
    lower_ci, upper_ci = np.percentile(bootstrap_samples_stat,[low_bound, high_bound])
    return lower_ci, upper_ci, bootstrap_samples_stat


def bootstrap_correlation_confidence_interval(a, v, stat_function=np.corrcoef, iterations=10000, ci=95):
    '''
    Bootstraps correlations using input confidence intervals.
    
    Parameters:
    a: Numpy array
        1-d numeric data.
        
    v: Numpy array
        1-d numeric data.
    
    stat_function: function, default="np.corrcoef"
        Function for calculating as sample statistic on data.
    
    iterations: int, default=10000
        Number of bootstrap samples to create.
    
    ci: int, default=95
        Percent of distribution encompassed by CI, 0<ci<100.
    '''
    sample1 = bootstrap(a, iterations=iterations)
    sample2 = bootstrap(v, iterations=iterations)
    bootstrap_samples_stat = list(map(stat_function, sample1, sample2))
    low_bound = (100. - ci) / 2
    high_bound = 100. - low_bound
    lower_ci, upper_ci = np.percentile(bootstrap_samples_stat,[low_bound, high_bound])
    return lower_ci, upper_ci, bootstrap_samples_stat


year_16_selected_variables = pd.read_spss('/Users/bn/Galvanize/GSS-Health-and-Happiness/data/GSS2016.sav', usecols=[
    'RHLTHEND', 
    'CESD1', 
    'CESD3', 
    'TWITTER', 
    'FACEBOOK', 
    'USUALHRS',
    'INTWKDYH', 
    'NOGO',  
    'OWNGUN', 
    'NEWS', 
    'TVHOURS', 
    'STRESSWK'
    ])

df = year_16_selected_variables.copy()


df['RHLTHEND'] = df['RHLTHEND'].astype(str).str.lower()
def convert_to_scale(val):
    answers = {'nan':-1,'poor': 1, 'fair': 2, 'good': 3, 'excellent':4}
    return answers.get(val)
df['health']=df['RHLTHEND'].map(convert_to_scale)
df.drop('RHLTHEND', axis=1, inplace=True)


df['CESD1'] = df['CESD1'].astype(str).str.lower()  
def convert_to_scale(val):
    answers = {'nan':-1,'none or almost none of the time': 1, 'some of the time': 2, 'most of the time': 3, 'all or almost all of the time':4}
    return answers.get(val)
df['felt_dep']=df['CESD1'].map(convert_to_scale)
df.drop('CESD1', axis=1, inplace=True)


df['CESD3'] = df['CESD3'].astype(str).str.lower()  
def convert_to_scale(val):
    answers = {'nan':-1,'none or almost none of the time': 1, 'some of the time': 2, 'most of the time': 3, 'all or almost all of the time':4}
    return answers.get(val)
df['felt_hap']=df['CESD3'].map(convert_to_scale)
df.drop('CESD3', axis=1, inplace=True)


df['TWITTER'] = df['TWITTER'].astype(str)  
def convert_to_scale(val):
    answers = {'nan':-1, 'No':True, 'Yes':False}
    return answers.get(val)
df['twitter']=df['TWITTER'].map(convert_to_scale)
df.drop('TWITTER', axis=1, inplace=True)


df['FACEBOOK'] = df['FACEBOOK'].astype(str)  
def convert_to_scale(val):
    answers = {'nan':-1, 'No':True, 'Yes':False}
    return answers.get(val)
df['facebook']=df['FACEBOOK'].map(convert_to_scale)
df.drop('FACEBOOK', axis=1, inplace=True)


df['NEWS'] = df['NEWS'].astype(str).str.lower() 
def convert_to_scale(val):
    answers = {'nan':-1, 'never': 1, 'less than once wk': 2, 'once a week': 3, 'few times a week':4, 'everyday':5}
    return answers.get(val)
df['news']=df['NEWS'].map(convert_to_scale)
df.drop('NEWS', axis=1, inplace=True)


df['OWNGUN'] = df['OWNGUN'].astype(str)  
def convert_to_scale(val):
    answers = {'nan':-1, 'No':0, 'Yes':1, 'REFUSED':.5}
    return answers.get(val)
df['own_gun']=df['OWNGUN'].map(convert_to_scale)
df.drop('OWNGUN', axis=1, inplace=True)


df['int_hrs_wkdy'] = df['INTWKDYH'].astype(float)  
df.drop('INTWKDYH', axis=1, inplace=True)


df['STRESSWK'] = df['STRESSWK'].astype(str).str.lower()  
def convert_to_scale(val):
    answers = {'nan':-1, 'almost no stress at all': 1, 'relatively little stress': 2, 'a moderate amount of stress': 3, 'a lot of stress':4}
    return answers.get(val)
df['stress']=df['STRESSWK'].map(convert_to_scale)
df.drop('STRESSWK', axis=1, inplace=True)


df['NOGO'] = df['NOGO'].astype(str)  
def convert_to_scale(val):
    answers = {'nan':-1, 'No':0, 'Yes':1}
    return answers.get(val, np.nan)
df['nogo']=df['NOGO'].map(convert_to_scale)
df.drop('NOGO', axis=1, inplace=True)


df['hrs_tv_day'] = df['TVHOURS'].astype(float)
tvmean = df['hrs_tv_day'].mean()  
df['TVHOURS'] = df['TVHOURS'].astype(str).str.lower()  
def convert_to_scale(val):
    answers = {'NaN':tvmean, '1.0': 1, '2.0': 2, '3.0': 3, '4.0':4,  '5.0': 5, '6.0': 2, '6.0': 3, '7.0':7, '8.0': 8, '9.0': 9, '10.0': 10, '11.0': 11, '12.0': 12, '13.0': 13, '14.0': 14, '15.0': 15, '16.0': 16, '17.0': 17, '18.0': 18, '19.0': 19, '20.0': 20, '21.0': 21, '22.0': 22, '23.0': 23, '24.0': 24}
    return answers.get(val)
df['hrs_tv_day']=df['TVHOURS'].map(convert_to_scale)
df.drop('TVHOURS', axis=1, inplace=True)


df['hrs_wrk_wk'] = df['USUALHRS'].astype(float)  
df.drop('USUALHRS', axis=1, inplace=True)


df = df.drop_duplicates()


correlations = correlations(df)
print(correlations)


corr_df = df[["health","felt_dep", "felt_hap"]]

corr_df['health'] = corr_df['health'].astype(float)
corr_df['felt_dep'] = corr_df['felt_dep'].astype(float)
corr_df['felt_hap'] = corr_df['felt_hap'].astype(float)


lst = [corr_df[corr_df['health']>0]['health'], corr_df[corr_df['felt_hap']>0]['felt_hap'], corr_df[corr_df['felt_dep']>0]['felt_dep']]

fig, ax = plt.subplots(figsize=(7,7))
ax.hist(lst, histtype='bar', label=['Health', 'Time felt happy', 'Time felt depressed'], color = ['green', 'gold', 'cornflowerblue'], edgecolor='k')
ax.set_title('Health Ratings and Time Felt Happy/Depressed', size=14, weight='bold', color='navy')
ax.set_xlabel('Health/Frequency of Feelings Ratings', size=12)
ax.set_ylabel('# of Respondents', size=12)
ax.legend(prop={'size': 9}, edgecolor='k')
ticks = [1,2,3,4]
labels = ['1', '2', '3', '4']
plt.xticks(ticks, labels)
plt.show()
plt.savefig('response_distributions')
plt.close(fig)


health_col = corr_df[corr_df['health']>0]['health']
hap_col = corr_df[corr_df['felt_hap']>0]['felt_hap']
dep_col = corr_df[corr_df['felt_dep']>0]['felt_dep']
health_ci = bootstrap_confidence_interval(health_col)

fig, ax = plt.subplots(figsize=(7,7))
ax.hist(health_ci[2], edgecolor='k', bins=20)
ax.axvline(health_ci[0], color='red')
ax.axvline(health_ci[1], color='red')
ax.set_title('Bootstrapped Means for 2016 Health Ratings', size=15, weight='bold', color='navy')
ax.set_xlabel('Health rating for past year', size=13)
ax.set_ylabel('# of Respondents', size=13)
plt.show()
plt.savefig('bootsrapped_health_ratings')
plt.close(fig)

felt_hap_ci = bootstrap_confidence_interval(hap_col)
fig, ax = plt.subplots(figsize=(7,7))
ax.hist(felt_hap_ci[2], edgecolor='k', bins=20)
ax.axvline(felt_hap_ci[0], color='red')
ax.axvline(felt_hap_ci[1], color='red')
ax.set_title('Bootstrapped Means for Time Felt Happy', size=15, weight='bold', color='navy')
ax.set_xlabel('Time felt happy in past year', size=14)
ax.set_ylabel('# of Respondents', size=14)
plt.show()
plt.savefig('bootstrapped_time_felt_happy')
plt.close(fig)

felt_dep_ci = bootstrap_confidence_interval(dep_col)
fig, ax = plt.subplots(figsize=(7,7))
ax.hist(felt_dep_ci[2], edgecolor='k', bins=20)
ax.axvline(felt_dep_ci[0], color='red')
ax.axvline(felt_dep_ci[1], color='red')
ax.set_title('Bootstrapped Means for Time Felt Depressed', size=15, weight='bold', color='navy')
ax.set_xlabel('Time felt depressed in past year', size=14)
ax.set_ylabel('# of Respondents', size=14)
plt.show()
plt.savefig('bootstrapped_time_felt_depressed')
plt.close(fig)


health_hap_corr = bootstrap_correlation_confidence_interval(df['health'],df['felt_hap'])
health_dep_corr = bootstrap_correlation_confidence_interval(df['health'],df['felt_hap'])

print(health_hap_corr)
print(health_dep_corr)
