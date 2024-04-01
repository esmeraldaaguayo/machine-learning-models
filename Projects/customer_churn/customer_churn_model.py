import pandas as pd
from customer_churn.utils import (
    generate_auc_curve, compose_confusion_matrix_plot, plot_feature_importance,
    data_cleanup, training_data_preparation
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# access data
train_df = data_cleanup("data/train.csv")
test_df = data_cleanup("data/test.csv")

x, y = training_data_preparation(train_df)

# split data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)
print(x_train.head())

# train random forest model
clf = RandomForestClassifier(max_depth=100, random_state=0, class_weight="balanced")
clf.fit(x_train, y_train)

# make predictions
y_pred = clf.predict(x_val)
probabilities = clf.predict_proba(x_val)[:, 1]

# evaluate model using validation data
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# visualize results
compose_confusion_matrix_plot(y_val, y_pred)
generate_auc_curve(y_val, probabilities)

# explainable AI
plot_feature_importance(x_train, clf)