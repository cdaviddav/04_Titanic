
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:39 2017

@author: cdavid
"""

""" Import all packages and the used settings and functions """

import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


from settings import Settings
from src.functions.data_preprocessing import *
from src.functions.functions_plot import *
from src.functions.classification_pipeline import *

settings = Settings()


""" ---------------------------------------------------------------------------------------------------------------
Load training and test dataset
"""

# Load train and test dataset
df_train = pd.read_csv(settings.config['Data Locations'].get('train'))
df_validate = pd.read_csv(settings.config['Data Locations'].get('test'))

df_train.name = 'df_train'
df_validate.name = 'df_validate'




""" ---------------------------------------------------------------------------------------------------------------
Explore the data
First take a look at the training dataset
- what are the features and how many features does the training data include
- are the missing values (but take a deeper look at the data preperation process)
- what are the different units of the features
"""

# Get a report of the training and test dataset as csv
# -> Use the function describe_report(df, name, output_file_path=None)
#describe_report(df_train, output_file_path=settings.csv)
#describe_report(df_validate, output_file_path=settings.csv)

# Show if there are different columns in the training and test dataset. If there is only one difference, it is likely, that its the target variable.
# If there are columns in the test dataset, which are not in the training dataset, they have to be deleted, because the algorithm will not see them during the training.
# -> Use the function column_diff(df_train, df_test)
column_diff(df_train, df_validate)

# Create boxplots to indentify outliers. Histograms are a good standard way to see if feature is skewed but to find outliers, boxplots are the way to use
# -> Use the function create_boxplots(df, output_file_path=None)
#create_boxplots(df_train, output_file_path=settings.figures)


#target_correlation(df=df_train, feature='Survived', k=5, output_file_path=settings.figures)



def analysis_Embarked(df_train, output_file_path=None):
    # Detailed analysis of Embarked
    df_train[df_train['Embarked'].isnull()] # 2 missing values with Pclass=1 and Fare=80$
    sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df_train)
    # We can see that for 1st class median line is coming around fare $80 for embarked value 'C'.
    # So we can replace NA values in Embarked column with 'C'
    df_train["Embarked"] = df_train["Embarked"].fillna("C")

    sns.factorplot('Embarked','Survived', data=df_train,size=4,aspect=3)

    fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
    sns.countplot(x='Embarked', data=df_train, ax=axis1)
    sns.countplot(x='Survived', hue="Embarked", data=df_train, order=[1,0], ax=axis2)

    # group by embarked, and get the mean for survived passengers for each value in Embarked
    embark_perc = df_train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
    sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'analysis_Embarked.pdf'))

analysis_Embarked(df_train, output_file_path=settings.figures) # if function is not used
# -> df_train["Embarked"] = df_train["Embarked"].fillna("C")



def analysis_Fare(df_train, df_validate):
    # Detailed analysis of Fare, only for test_df, since there is a missing "Fare" values
    df_validate[df_validate['Fare'].isnull()]
    # take the meadian of all fares of those passengers who share 3rd Passenger class and Embarked from 'S'
    median_fare = df_validate[(df_validate['Pclass'] == 3) & (df_validate['Embarked'] == 'S')]['Fare'].median()
    df_validate["Fare"] = df_validate["Fare"].fillna(median_fare)

    # convert from float to int
    df_train['Fare'] = df_train['Fare'].astype(int)
    df_validate['Fare'] = df_validate['Fare'].astype(int)

    # get fare for survived & didn't survive passengers
    fare_not_survived = df_train["Fare"][df_train["Survived"] == 0]
    fare_survived = df_train["Fare"][df_train["Survived"] == 1]

    # get average and std for fare of survived/not survived passengers
    avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])

    df_train['Fare'].plot(kind='hist', figsize=(15,3),bins=200, xlim=(0,50), title='Histogram of Fare')
    avgerage_fare.index.names = std_fare.index.names = ["Survived"]
    avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

analysis_Fare(df_train, df_validate)



def analysis_Pclass(df_train, output_file_path=None):
    grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    if output_file_path is not None:
        grid.savefig(os.path.join(output_file_path, 'analysis_Pclass.pdf'))

#analysis_Pclass(df_train, output_file_path=settings.figures)


""" ---------------------------------------------------------------------------------------------------------------
Feature Creation
"""

df_train['Family'] = df_train["Parch"] + df_train["SibSp"]
df_train['Family'].loc[df_train['Family'] > 0] = 1
df_train['Family'].loc[df_train['Family'] == 0] = 0

df_validate['Family'] = df_validate["Parch"] + df_validate["SibSp"]
df_validate['Family'].loc[df_validate['Family'] > 0] = 1
df_validate['Family'].loc[df_validate['Family'] == 0] = 0

# drop Parch & SibSp
df_train = df_train.drop(['SibSp','Parch'], axis=1)
df_validate = df_validate.drop(['SibSp','Parch'], axis=1)


def analysis_Family(df_train, output_file_path=None):
    fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
    sns.countplot(x='Family', data=df_train, order=[1,0], ax=axis1)
    # average of survived for those who had/didn't have any family member
    family_perc = df_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
    sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
    axis1.set_xticklabels(["With Family","Alone"], rotation=0)

    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'analysis_Family.pdf'))

#analysis_Family(df_train)




import re
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

titles_train = df_train["Name"].apply(get_title)
titles_validate = df_validate["Name"].apply(get_title)

df_train["Title"] = titles_train
df_validate["Title"] = titles_validate

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don',
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

# Reassign mlle, ms, and mme accordingly
def rename_title(df):
    df.loc[df["Title"] == "Mlle", "Title"] = 'Miss'
    df.loc[df["Title"] == "Ms", "Title"] = 'Miss'
    df.loc[df["Title"] == "Mme", "Title"] = 'Mrs'
    df.loc[df["Title"] == "Dona", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Lady", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Countess", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Capt", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Col", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Don", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Major", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Rev", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Sir", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Jonkheer", "Title"] = 'Rare Title'
    df.loc[df["Title"] == "Dr", "Title"] = 'Rare Title'
    df["Title"].value_counts()
    return df

df_train = rename_title(df_train)
df_validate = rename_title(df_validate)



df_train = labelEnc(df_train)
df_validate = labelEnc(df_validate)

from sklearn.ensemble import RandomForestRegressor
# predicting missing values in age using Random Forest
def fill_missing_age(df):
    # Feature set
    age_df = df[['Age', 'Embarked', 'Fare', 'Title', 'Pclass']]
    # Split sets into train and test
    train = age_df.loc[(df.Age.notnull())]  # known Age values
    test = age_df.loc[(df.Age.isnull())]  # null Ages

    # All age values are stored in a target array
    y = train.values[:, 0]

    # All the other values are stored in the feature array
    X = train.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df

df_train = fill_missing_age(df_train)
df_validate = fill_missing_age(df_validate)


df_train, df_validate = drop_missing_values(df_train, df_validate, limit=None, output_file_path=settings.csv)



def analysis_age(df_train, output_file_path=None):
    # peaks for survived/not survived passengers by their age
    facet = sns.FacetGrid(df_train, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, df_train['Age'].max()))
    facet.add_legend()

    if output_file_path is not None:
        facet.savefig(os.path.join(output_file_path, 'analysis_age_1.pdf'))


    # average survived passengers by age
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    average_age = df_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
    sns.barplot(x='Age', y='Survived', data=average_age)

    if output_file_path is not None:
        fig.savefig(os.path.join(output_file_path, 'analysis_age_2.pdf'))


#analysis_age(df_train, output_file_path=settings.figures)






""" ---------------------------------------------------------------------------------------------------------------
Feature Selection
"""


df_train.drop(['PassengerId', 'Name'], axis=1, inplace=True)
df_validate.drop(['PassengerId', 'Name'], axis=1, inplace=True)







""" ---------------------------------------------------------------------------------------------------------------
Machine Learning (Classification)
"""

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)

y_train = train_set['Survived']
x_train = train_set.drop(['Survived'], axis=1)

y_test = test_set['Survived']
x_test = test_set.drop(['Survived'], axis=1)

x_validate = df_validate

pipeline = Pipeline([
    ('reduce_dim', PCA()),
    ('feature_scaling', MinMaxScaler()), # scaling because linear models are sensitive to the scale of input features
    ('classification', RandomForestClassifier()),
    ])

param_grid = [{'reduce_dim__n_components': [5, 8],
               'classification__n_estimators': [20, 50, 100],
               'classification__max_depth': [3, 5, 10],
               'classification__min_samples_split': [2, 5]
              }]


pipe_best_params = classification_pipeline(x_train, y_train, pipeline, 5, 'accuracy', param_grid)

pipe_best = Pipeline([
    ('reduce_dim', PCA(n_components = pipe_best_params['reduce_dim__n_components'])),
    ('feature_scaling', MinMaxScaler()),
    ('regression', RandomForestClassifier(
        n_estimators = pipe_best_params['classification__n_estimators'],
        max_depth = pipe_best_params['classification__max_depth'],
        min_samples_split = pipe_best_params['classification__min_samples_split'],))
])

print(pipe_best_params['reduce_dim__n_components'])
print(pipe_best_params['classification__n_estimators'])
print(pipe_best_params['classification__max_depth'])
print(pipe_best_params['classification__min_samples_split'])

train_errors = evaluate_pipe_best_train(x_train, y_train, pipe_best, 'RandomForestClassifier')


plot_learning_curve(pipe_best, 'RandomForestClassifier', x_train, y_train, 'accuracy', output_file_path=settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Evaluate the System on the Test Set
"""
#Evaluate the model with the test_set
# -> Use the function evaluate_pipe_best_test(x_train, y_train, pipe_best, algo, output_file_path=None)
evaluate_pipe_best_test(x_test, y_test, pipe_best)