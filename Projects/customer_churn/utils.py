import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score

from sklearn.tree import export_graphviz
from IPython.display import Image, display
import graphviz
import pydot

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")


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


def compose_confusion_matrix_plot(cm):
    f, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig("plots/confusion_matrix.png")


def generate_auc_curve(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    auc_score = roc_auc_score(y, scores)

    f, ax = plt.subplots()
    sns.lineplot(x=fpr, y=tpr)
    ax.set_ylabel("tpr")
    ax.set_xlabel("fpr")
    ax.set_title(f"AUROC: {auc_score}")
    plt.savefig("plots/auc.png")


def plot_summary_scatterplots(df):
    f, axs = plt.subplots()
    f = sns.PairGrid(df)
    f.map_diag(sns.histplot)
    f.map_offdiag(sns.scatterplot)
    plt.savefig("plots/scatterplots_summary.png")