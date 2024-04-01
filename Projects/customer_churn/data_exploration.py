import pandas as pd
from customer_churn.utils import (
    plot_feature_correlation_matrix, plot_summary_histplots, plot_summary_violinplots,
    plot_target_counts, plot_summary_scatterplots
)

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

# visualize data
continuous_notarget_df = continuous_cols_df.drop(columns=["target"])
plot_feature_correlation_matrix(continuous_notarget_df)
plot_summary_scatterplots(continuous_notarget_df)
plot_summary_histplots(categorical_cols_df)
plot_summary_violinplots(continuous_cols_df)
plot_target_counts(continuous_cols_df)





