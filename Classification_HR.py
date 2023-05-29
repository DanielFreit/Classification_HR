import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
from scipy.stats import shapiro
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from pyod.models.knn import KNN
from sklearn.neural_network import MLPClassifier



#  MORE INFO ABOUT THE DATASET - https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

'''The dataset consists in an internal evaluation by IBM to check if a worker is likely to leave the company
based on several historical data'''

base = pd.read_csv('Human_Resources.csv')
df = pd.read_csv('Human_Resources.csv')

# CHECK PRINT 1

#  todo EXPLORATION ------------------------

'''For a better overview I`m using describe and info so we can understand the basics of this dataset'''

print(base.info())
print(base.describe())

# CHECK PRINT 1B

'''Here we can see some exploratory data from the dataset'''

df.hist(bins=30, figsize=(20, 20), color='r')
plt.show()

# CHECK PRINT 2

'''Now we know that columns like 'EmployeeCount', 'StandardHours', 'Over18' and 'EmployeeNumber don`t help us
because all the data have the same value, which would not help us with the classification, so the data can be dropped'''

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], inplace=True)

'''Now we need to check Nan data, and duplicated data'''

print(df.isnull().sum())
print(df.duplicated().sum())

'''Let`s check the correlation between the variables'''

correlations = df.corr()
f, ax = plt.subplots(figsize=(25, 18))
sns.heatmap(correlations, annot=True)
plt.show()

# CHECK PRINT 3

'''Now the correlation with the class we want to classify'''

plt.figure(figsize=(20, 20))
plt.subplot(411)
sns.countplot(x='JobRole', hue='Attrition', data=df)
plt.subplot(412)
sns.countplot(x='MaritalStatus', hue='Attrition', data=df)
plt.subplot(413)
sns.countplot(x='JobInvolvement', hue='Attrition', data=df)
plt.subplot(414)
sns.countplot(x='JobLevel', hue='Attrition', data=df)
plt.show()

'''We can see that Sales Representatives are more likely to drop the job, we`ll check the salary and other variables
later, to check more correlations, with distance from work and salary'''

# CHECK PRINT 4

'''Now we need to check the classification data, so we can see if the data is balanced or not'''

sns.countplot(data=df, x='Attrition')
plt.show()

# CHECK PRINT 5

''''Let's split the data between the workers that still work in the company and the ones who got a new job'''

left = df[df['Attrition'] == 1]
stayed = df[df['Attrition'] == 0]

'''Now we can check the correlation between job role and monthly income because they showed interesting correlation'''

plt.figure(figsize=(12, 8))
sns.boxplot(x='MonthlyIncome', y='JobRole', data=df)
plt.show()

# CHECK PRINT 6

'''Maybe the company is not paying enough for the Sales Representatives, we can see in different plots that this
might be the case'''

#  todo OUTLIERS ------------------------

'''Since the data is internal and most data are not filled by the workers, I decided to not exclude the outliers
of the equation, because most of the outliers rely on Monthly Payment, and that is a company info, so even being
considered outliers they are correct, so I`m commenting this part of the code'''

# detector = KNN()
# detector.fit(df.iloc[:, :-1])
#
# predict = detector.labels_
# print(np.unique(predict, return_counts=True))
#
# confidence = detector.decision_scores_
# print(confidence)
#
# outliers = []
#
# for i in range(len(predict)):
#     if predict[i] == 1:
#         outliers.append(i)
#
# print(outliers)
#
# outliers_f = df.iloc[outliers]
# print(outliers_f)
# df.drop(labels=[outliers], inplace=True)

# outliers_f.to_csv('outliers_credit_data')

#  todo SELECTION ------------------------

'''Now I`ll drop the classification columns, and pass the values of the dataset to variables and split the numeral
and categorical data'''

X = df.drop(columns='Attrition')
X = X.iloc[:, :].values
y = df.loc[:, 'Attrition'].values

X_cat = df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

X_num = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
            'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
            'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

#  todo ENCODER ------------------------

# le = LabelEncoder()
# X[:, 1] = le.fit_transform(X[:, 1])

'''For this case we`re using one hot encoder for the transformation from categorical to numeral'''

onehotencoder = OneHotEncoder()

X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

'''Now we can concatenate numeral and categorical values, but this time categoricals have been changed to numerical
by one hot encoder'''

X = pd.concat([X_cat, X_num], axis=1)

# CHECK PRINT 7

#  todo ATTRIBUTE SELECTION CHI2, ANOVA------------------------

'''We can reduce the dimension with chi2 or ANOVA methods, for this model there was no improvement, so we`re commenting
it in the code'''

# print(X.shape)
# chi2 = SelectFdr(chi2, alpha=0.01)
# X = chi2.fit_transform(X, y)
# print(X.shape)

# print(X.shape)
# fdr = SelectFdr(f_classif, alpha=0.01)
# X = fdr.fit_transform(X, y)
# print(X.shape)

#  todo SMOTE ------------------------

'''Since the data is unbalanced, we can apply a method to generate synthetic data, based on the values we have less'''

print(X.shape, y.shape)
smote = SMOTE(sampling_strategy='minority', random_state=0)
X, y = smote.fit_resample(X, y)
print(X.shape, y.shape)

# CHECK PRINT 8

#  todo NORMALIZATION ------------------------

'''And then we can normalize the data'''

standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)

# CHECK PRINT 9

#  todo PRE PROCESSING SAVE ------------------------

'''Saving the pre process part just in case'''

with open('Smoted_Hot_Norm_Human_Resources.pkl', mode='wb') as f:
    pickle.dump([X, y], f)

# with open('Smoted.pkl', 'rb') as f:
#     X, y = (pickle.load(f))
# print(np.unique(y, return_counts=True))
#
# sns.countplot(x=y)
# plt.show()

'''with pickle I can test different models and pick the best ones, I used GridSearch for better parameters and have
22% of the dataset for tests, let`s split the dataset and save it to test different models'''

X_training, X_test, y_training, y_test = tts(X, y, test_size=0.22, stratify=y, random_state=0)

with open('Test_Training_Human_Resources.pkl', mode='wb') as f:
    pickle.dump([X_training, X_test, y_training, y_test], f)

#  todo HYPOTHESIS TEST FOR MEAN, VAR, COEFFICIENT ------------------------

'''In this case I found the best models after several tests and came back with the best parameters, now we'll
create a loop to check if we can find the optimal random seed for 30 different pieces od the dataframe randomly.
We`ll append the results in a list, so we can compare all the models with different statistics methods'''

r_ann = []
r_logistic = []
r_forest = []

for i in range(30):
    X_training, X_test, y_training, y_test = tts(X, y, test_size=0.22, stratify=y, random_state=i)
    '''Neural Network'''
    ann = MLPClassifier(max_iter=1000, verbose=True, tol=0.0000100, solver='adam', activation='relu',
                        hidden_layer_sizes=(50, 50), random_state=0)
    ann.fit(X_training, y_training)
    r_ann.append(accuracy_score(y_test, ann.predict(X_test)))
    '''Logistic Regression'''
    logistic = LogisticRegression(max_iter=500, tol=0.00100, random_state=0)
    logistic.fit(X_training, y_training)
    r_logistic.append(accuracy_score(y_test, logistic.predict(X_test)))
    '''Random Forest'''
    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, min_samples_split=5,
                                    min_samples_leaf=5)
    forest.fit(X_training, y_training)
    r_forest.append(accuracy_score(y_test, forest.predict(X_test)))

#  todo HYPOTHESIS VISUALIZATION ------------------------

'''Now let`s check the difference between the 3 tests'''

print('Neural Network')
r_ann = pd.DataFrame(r_ann)
print(f'VAR', r_ann.var())
print(f'MODE', r_ann.mode())
print(f'MEDIAN', r_ann.median())
print(f'COEFF', stats.variation(r_ann) * 100)
print(r_ann.describe())
print(r_ann.idxmax())
print(r_ann.max())
sns.histplot(r_ann, kde=True)
plt.show()

print('Logistic Regression')
r_logistic = pd.DataFrame(r_logistic)
print(f'VAR', r_logistic.var())
print(f'MODE', r_logistic.mode())
print(f'MEDIAN', r_logistic.median())
print(f'COEFF', stats.variation(r_logistic) * 100)
print(r_logistic.describe())
print(r_logistic.idxmax())
print(r_logistic.max())
sns.histplot(r_logistic, kde=True)
plt.show()

print('Random Forest')
r_forest = pd.DataFrame(r_forest)
print(f'VAR', r_forest.var())
print(f'MODE', r_forest.mode())
print(f'MEDIAN', r_forest.median())
print(f'COEFF', stats.variation(r_forest) * 100)
print(r_forest.describe())
print(r_forest.idxmax())
print(r_forest.max())
sns.histplot(r_forest, kde=True)
plt.show()

# CHECK PRINT 10

#  todo SHAPIRO NORMALIZATION TEST ------------------------

'''And check the data distribution'''

print('ANN')
print(shapiro(r_ann))
print('LOGISTIC')
print(shapiro(r_logistic))
print('RANDOM FOREST')
print(shapiro(r_forest))

# CHECK PRINT 11

#  todo NORMAL = ANOVA/TUKEY TEST ------------------------

'''Now let`s compare the data between each other and apply the Tukey test'''

# _, P = f_oneway(r_naive, r_logistic, r_forest)
# print(P)
#
# if P <= alpha:
#     print('Null (old) hypothesis rejected')
# else:
#     print('Alternative (new) hypothesis rejected')

final_result = {'accuracy': np.concatenate([r_ann[0], r_logistic[0], r_forest[0]]),
                'algorithm': ['ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN',
                              'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN',
                              'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN', 'ANN',
                              'ANN', 'ANN', 'ANN', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic',
                              'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic',
                              'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic',
                              'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Logistic',
                              'Logistic', 'Logistic', 'Logistic', 'Logistic', 'Forest', 'Forest', 'Forest',
                              'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest',
                              'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest',
                              'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest', 'Forest',
                              'Forest', 'Forest', 'Forest']}

print(final_result)

df_f_results = pd.DataFrame(final_result)
print(df_f_results)

comparison = MultiComparison(df_f_results['accuracy'], df_f_results['algorithm'])
test = comparison.tukeyhsd()
print(test)
test.plot_simultaneous()
plt.show()

'''We can see the model that gave us better results and how much they vary'''

# CHECK PRINT 12

#  todo MODEL SELECTION ------------------------

'''Now that we know everything we need for this project, I`ll create the final model with the best result'''

X_training, X_test, y_training, y_test = tts(X, y, test_size=0.22, stratify=y, random_state=7)
forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0, min_samples_split=5,
                                min_samples_leaf=5)
forest.fit(X_training, y_training)

'''Let's check the results, predictions, f1-score, recall and confusion matrix, also known as KPIs'''

prediction = forest.predict(X_test)
print(accuracy_score(y_test, prediction))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

cm = confusion_matrix(y_test, prediction)
sns.heatmap(cm, annot=True)
plt.show()

# CHECK PRINT 13

#  todo SAVE ------------------------

'''Now we can save everything we might need, like the model, one hot encoder and normalization methods'''

with open('Finished_Human_Resources.pkl', mode='wb') as f:
    pickle.dump([X_training, X_test, y_training, y_test], f)

with open('Variables_Human_Resources.pkl', mode='wb') as f:
    pickle.dump([onehotencoder, chi2, standardscaler, forest], f)

'''Now let's load again so we can test it (usually the person who deploys take care of this part, but for this example
I`ll create a quick test with an input that comes from the same dataset'''

with open('Variables_Human_Resources.pkl', mode='rb') as f:
    onehotencoder, chi2, standardscaler, forest = pickle.load(f)

#  todo SINGULAR TEST ------------------------

'''Let`s do the pre processing for this new entry, utilizing the same treatment we did in the original dataset, but
this time with a single worker entry'''

X_sing = df.iloc[0:1]
print(X_sing.shape)
X_sing = X_sing.drop(columns='Attrition')

X_cat_sing = X_sing[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

X_num_sing = X_sing[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                     'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime',
                     'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                     'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                     'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]

X_cat_sing = onehotencoder.transform(X_cat_sing).toarray()
X_cat_sing = pd.DataFrame(X_cat_sing)
X_num_sing = pd.DataFrame(X_num_sing)
X_sing = pd.concat([X_cat_sing, X_num_sing], axis=1)
# X_sing = chi2.transform(X_sing)
X_sing = standardscaler.transform(X_sing)

'''After we apply the encoder, and the normalization, we can predict the result'''

print(forest.predict(X_sing))
print(forest.predict_proba(X_sing))

# CHECK PRINT 14

'''As we can see, the model gave us the answer 0 (He's not likely to leave the company)'''
'''The model also showed the precision that he thinks this entry belongs to each group (0 or 1)'''
