import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# import random forest from sklearn
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
file_path = 'diabetes_binary_health_indicators_BRFSS2015.csv'
df = pd.read_csv(file_path)

# rename Diabetes_binary to Diabetes
df.rename(columns = {'Diabetes_binary' : 'Diabetes'}, inplace = True)

# # Use features decided on
# prune_features = ['Diabetes']
# X = df.drop(columns=prune_features)
# y = df['Diabetes']

# # split into testing and training set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# # further split into training and validation set
# X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size = 0.20, random_state = 1)

# # use SMOTE oversampling for preprocessing, data was imbalanced
# smote = SMOTE(random_state=42)
# X_smote, y_smote = smote.fit_resample(X_train, y_train)

# # run gridsearchcv with various n_estimators on random forest

# # create model
# model = RandomForestClassifier(random_state=42)

# # create dict of hyperparams to tune
# hyperparam_dict = {
#     'n_estimators': [50, 100, 150, 200]
# }

# print('start grid search')

# # create gridsearchcv object
# grid = GridSearchCV(model, hyperparam_dict, cv=5, n_jobs=-1)

# # fit gridsearchcv object
# grid.fit(X_smote, y_smote)

# # get best hyperparams
# best_hyperparams = grid.best_params_
# print(best_hyperparams)

# # get best model
# best_model = grid.best_estimator_

# # evaluate model
# best_model.fit(X_smote, y_smote)
# y_pred = best_model.predict(X_test)

# # print classification report
# print(classification_report(y_test, y_pred))

'''
The features that correlated better with the target are: HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, GenHlth , MentHlth, PhysHlth, DiffWalk, Age, Education, Income.

The model with best performance for the unseen balanced dataset is the Random Forest Model with SMOTE undersampling.
'''

# Use the features that correlated better with the target: HighBP, HighChol, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, GenHlth , MentHlth, PhysHlth, DiffWalk, Age, Education, Income.
prune_features = ['Diabetes', 'CholCheck', 'Fruits', 'Veggies', 'AnyHealthcare', 'NoDocbcCost',
       'Sex']
X = df.drop(columns=prune_features)
y = df['Diabetes']

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# further split into training and validation set
X_train2, X_valid, y_train2, y_valid = train_test_split(X_train, y_train, test_size = 0.20, random_state = 1)

n_estimators = [50, 100, 150, 200]



# use SMOTEENN oversampling on training set - default oversampling
smoteenn = SMOTEENN(random_state=42)
X_smote, y_smote = smoteenn.fit_resample(X_train2, y_train2)

# Use RandomUnderSampler to balance the dataset - default undersampling
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train2, y_train2)

# random undersample the dataset, then oversample the result using smoteenn
rus2 = RandomUnderSampler(sampling_strategy=.6, random_state=42)
X_rus2, y_rus2 = rus2.fit_resample(X_train2, y_train2)
X_smote2, y_smote2 = smoteenn.fit_resample(X_rus2, y_rus2)

# Split X_train2, y_train2 into oversampled and undersampled datasets, oversample and undersample each dataset, then combine the results
X_oversample, X_undersample, y_oversample, y_undersample = train_test_split(X_train2, y_train2, test_size = 0.5, random_state = 1)
# use SMOTEENN on half of the training data
X_smote3, y_smote3 = smoteenn.fit_resample(X_oversample, y_oversample)
# use RandomUnderSampler on half of the training data
rus3 = RandomUnderSampler(random_state=42)
X_rus3, y_rus3 = rus3.fit_resample(X_undersample, y_undersample)
# combine oversampled and undersampled datasets
X_final = pd.concat([X_smote3, X_rus3])
y_final = pd.concat([y_smote3, y_rus3])

x_methods = [X_smote, X_rus, X_smote2, X_final]
y_methods = [y_smote, y_rus, y_smote2, y_final]

# perform validation hyperparameter tuning on random forest using hyperparam_dict.
model = RandomForestClassifier(random_state=42)

# create dict of results
results = {}

for i in range(len(x_methods)):
    print(f"\n\nMethod {i}\n\n")
    for n in n_estimators:
        # create model with n_estimators and balanced class weights balanced
        model = RandomForestClassifier(n_estimators=n, random_state=42, class_weight='balanced_subsample')
        # fit model with oversampled training data
        model.fit(x_methods[i], y_methods[i])
        # predict on validation set
        y_pred = model.predict(X_valid)
        # store classification report in results
        class_report = classification_report(y_valid, y_pred)
        print(f"n_estimators: {n}")
        print(class_report)
        print()

# print results
# for n in results:
#     print(f"n_estimators: {n}")
#     print(results[n])
#     print()




'''
Add data cleaning - remove duplicate data entries and change columns to uint8

# list of columns used from first exp (BMI group). Spoiler: better score for unseen dataset
cols_list = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump',
       'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Education',
       'Income']

#scaler
scaler = StandardScaler()
scaler.fit(X)

test that all are not diabetic and show results. Shows why recall is important instead of accuracy

X_train, X_test, y_train, y_test = train_test_split(
    X, # predictive variables
    y, # target
    test_size=0.1, # portion of dataset to allocate to test set
    random_state=508312, # we are setting the seed here
)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


'''