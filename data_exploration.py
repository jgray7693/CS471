import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'diabetes_binary_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())          # View first few rows
print(data.info())          # Data types and non-null counts
print(data.describe())      # Summary statistics
print(data.isnull().sum())  # Check for missing values


# # Class distribution
# sns.countplot(x='Diabetes_binary', data=data)
# plt.title('Class Distribution: Diabetes vs No Diabetes')
# plt.xlabel('0 = No Diabetes, 1 = Diabetes')
# plt.ylabel('Count')
# plt.show()

# # Print percentage distribution
# class_counts = data['Diabetes_binary'].value_counts(normalize=True) * 100
# print(f"Class Distribution:\n{class_counts}")



# Continuous features
continuous_features = ['BMI', 'Age', 'HighBP', 'Smoker', 'Education', 'Income']

# Boxplot for continuous features by target
for feature in continuous_features:
    plt.figure()
    sns.countplot(x=feature, hue='Diabetes_binary', data=data)
    plt.title(f'{feature} by Diabetes Status')
    plt.xlabel('0 = No Diabetes, 1 = Diabetes')
    plt.ylabel(feature)
    plt.show()



# # Correlation matrix
# plt.figure(figsize=(12, 8))
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Feature Correlation Heatmap')
# plt.show()

# # Highly correlated features
# threshold = 0.8
# high_corr_pairs = correlation_matrix[correlation_matrix.abs() > threshold]


# Pairplot for selected features
# selected_features = ['BMI', 'Age', 'PhysActivity', 'Smoker', 'Diabetes_binary']
# sns.pairplot(data[selected_features], hue='Diabetes_binary', diag_kind='kde')
# plt.show()



# Binary categorical features
binary_features = ['Smoker', 'HighBP', 'HighChol', 'GenHlth']

for feature in binary_features:
    plt.figure()
    sns.countplot(x=feature, hue='Diabetes_binary', data=data)
    plt.title(f'{feature} Distribution by Diabetes Status')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(['No Diabetes', 'Diabetes'])
    plt.show()



