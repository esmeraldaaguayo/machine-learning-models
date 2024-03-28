import pandas as pd
from customer_churn.utils import (
    plot_feature_correlation_matrix, plot_summary_histplots, plot_summary_violinplots,
    generate_one_hot_columns, plot_target_counts, generate_auc_curve,
    compose_confusion_matrix_plot, plot_summary_scatterplots
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt

# load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# data cleanup
train_df.rename(str.lower, axis='columns', inplace=True)
train_df = train_df.rename(columns={col: col.replace(' ', '_') for col in train_df.columns})
train_df = train_df.rename(columns={col: col.replace('-', '_') for col in train_df.columns})

test_df.rename(str.lower, axis='columns', inplace=True)
test_df = test_df.rename(columns={col: col.replace(' ', '_') for col in train_df.columns})
test_df = test_df.rename(columns={col: col.replace('-', '_') for col in train_df.columns})

# explore data
print(f"training dataset shape: {train_df.shape}")
categorical_cols_df = train_df.copy().select_dtypes(include=['object', 'category'])
continuous_cols_df = train_df.drop(columns=categorical_cols_df.columns)
# print(categorical_cols_df.shape)
# print(categorical_cols_df.value_counts())

# visualize data
continuous_notarget_df = continuous_cols_df.drop(columns=["target"])
plot_feature_correlation_matrix(continuous_notarget_df)
plot_summary_scatterplots(continuous_notarget_df)
plot_summary_histplots(categorical_cols_df)
plot_summary_violinplots(continuous_cols_df)
plot_target_counts(continuous_cols_df)

# split out validation data
x = train_df.drop(columns=['target'])
y = train_df['target']

# remove unusable features
x = x.drop(columns=['id'])
cols_to_onehot = [col for col in categorical_cols_df.columns if col != 'id']

# convert categorical to numerical values
x = generate_one_hot_columns(x, cols_to_onehot)

# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

# train random forest model
clf = RandomForestClassifier(max_depth=100, random_state=0, class_weight="balanced")
clf.fit(x_train, y_train)

# make predictions
y_pred = clf.predict(x_val)

# evaluate model using validation data
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y_val, y_pred)


# visualize results
compose_confusion_matrix_plot(cm)
generate_auc_curve(y_val, y_pred)



