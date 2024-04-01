import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt
# sns.set_style("white")
sns.set_style('darkgrid')


def plot_feature_correlation_matrix(df):
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title("correlation summary")
    plt.savefig("plots/correlation_summary.png")


def plot_summary_histplots(df):
    # remove specific columns
    df = df.drop(columns=["connect_date", "tariff_ok", "id"])

    f, axs = plt.subplots(3, 2, figsize=(20, 10))
    # Flatten the 3x2 subplots into a 1D array for easier iteration
    axs = axs.flatten()
    # Create 6 plots and populate the subplots
    columns_to_plot = list(df.columns)
    for i, ax in enumerate(axs):
        # Example plot
        sns.histplot(data=df, x=columns_to_plot[i], ax=ax)
    plt.tight_layout()
    plt.savefig("plots/categorical_barplots_summary.png")


def plot_summary_violinplots(df):
    f, axs = plt.subplots(6, 5, figsize=(40, 20))
    # Flatten the 6x5 subplots into a 1D array for easier iteration
    axs = axs.flatten()
    # Create plots and populate the subplots
    columns_to_plot = list(df.columns)
    for i, ax in enumerate(axs):
        # Example plot
        sns.violinplot(data=df, y=columns_to_plot[i], ax=ax)
    plt.tight_layout()
    plt.savefig("plots/continuous_distributions_summary.png")


def plot_target_counts(df):
    f, ax = plt.subplots()
    sns.barplot(y=df.target.value_counts().tolist(), x=df.target.value_counts().index)
    ax.set_ylabel("counts")
    ax.set_title("Training data labels for customer churn prediction")
    plt.savefig("plots/target_counts_summary.png")


def generate_one_hot_columns(df, cat_cols):
    for col in cat_cols:
        # Get one hot encoding of columns
        one_hot = pd.get_dummies(df[col], prefix=col, dtype='int')
        # remove original column
        df = df.drop(columns=[col], axis=1)
        # Join the encoded df
        df = df.join(one_hot)
    return df


def compose_confusion_matrix_plot(y_val, y_pred):
    f, ax = plt.subplots()
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, fmt="d", annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.savefig("plots/confusion_matrix.png")


def generate_auc_curve(y, probabilities):
    fpr, tpr, thresholds = roc_curve(y, probabilities, pos_label=1)
    auc_score = roc_auc_score(y, probabilities)

    f, ax = plt.subplots()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label="Auc : %.3f" % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig("plots/auc.png")


def plot_feature_importance(x, classifier):
    features = pd.DataFrame({"features": x.columns.tolist()})
    importance = pd.DataFrame(classifier.feature_importances_).rename(columns={0: "importance"})

    coef_sumry = pd.concat([features, importance], axis=1)
    coef_sumry = coef_sumry.sort_values(by="importance", ascending=False)
    coef_sumry = coef_sumry.head(10)

    f, ax = plt.subplots()
    sns.barplot(x=coef_sumry["features"], y=coef_sumry["importance"])
    plt.title('Feature Importance')
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")


def plot_summary_scatterplots(df):
    f, axs = plt.subplots()
    f = sns.PairGrid(df)
    f.map_diag(sns.histplot)
    f.map_offdiag(sns.scatterplot)
    plt.savefig("plots/scatterplots_summary.png")


def data_cleanup(filepath):
    df = pd.read_csv(filepath)
    df.drop(["id"], inplace=True, axis=1)

    # handle null values
    df.replace(" ", np.nan)
    df.fillna(0, inplace=True)

    # standardize column format
    df.rename(str.lower, axis='columns', inplace=True)
    df = df.rename(columns={col: col.replace(' ', '_') for col in df.columns})
    df = df.rename(columns={col: col.replace('-', '_') for col in df.columns})
    return df


def training_data_preparation(df):
    # split out validation data
    x = df.drop(columns=['target'])
    y = df['target']

    # convert categorical to numerical values
    cols_to_onehot = df.copy().select_dtypes(include=['object', 'category']).columns
    x = generate_one_hot_columns(x, cols_to_onehot)
    return x, y