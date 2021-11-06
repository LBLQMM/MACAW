# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:23:53 2021

Contains plotting functions for MACAW project.

@author: Vincent
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import auc


# ----- Plotting functions -----


def parity_plot(
    x,
    y,
    x_test=None,
    y_test=None,
    y_train_std=None,
    y_test_std=None,
    xlabel='True value',
    ylabel='Predicted',
    title=None,
    savetitle=None,
):
    x = np.array(x)
    y = np.array(y)

    # Plot Figures
    plt.figure(figsize=(5.5, 5.0))

    # Find the boundaries of X and Y values
    if x_test is not None:
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        min1 = min(x.min(), y.min(), x_test.min(), y_test.min())
        max1 = max(x.max(), y.max(), x_test.max(), y_test.max())

    else:
        min1 = min(x.min(), y.min())
        max1 = max(x.max(), y.max())
    rng1 = max1 - min1
    bounds = (min1 - 0.05 * rng1, max1 + 0.05 * rng1)

    # Reset the limits
    ax = plt.gca()
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)

    # plot the diagonal
    ax.plot([0, 1], [0, 1], 'k-', lw=0.5, transform=ax.transAxes)

    # Plot the data
    plt.errorbar(x=x, y=y, yerr=y_train_std, color='blue', fmt='.', elinewidth=.5)

    if x_test is not None:
        plt.errorbar(x=x_test, y=y_test, yerr=y_test_std, color='red', fmt='.', elinewidth=.5)
        text1 = f"$R^2_{{train}} = {r2_score(x,y):0.2f}$"
        text2 = f"$R^2_{{test}} = {r2_score(x_test,y_test):0.2f}$"

        plt.gca().text(0.05, 0.93, text1, transform=plt.gca().transAxes, c='blue')
        plt.gca().text(0.05, 0.85, text2, transform=plt.gca().transAxes, c='red')

    else:
        text1 = f"$R^2= {r2_score(x,y):0.2f}$"
        plt.gca().text(0.05, 0.93, text1, transform=plt.gca().transAxes)

    # Title and labels
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the figure into 300 dpi
    if savetitle:
        plt.savefig(savetitle, format='png', dpi=300, bbox_inches='tight',
                    treansparent=False)
    else:
        plt.show()


def plot_precision_vs_recall(precisions, recalls, precisions_test=None, recalls_test=None, title=None, savetitle=None):
    fignow = plt.figure(figsize=(6.5, 5.5))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.axis([0, 1, 0, 1])

    if precisions_test is None:
        text1 = f"$AUPRC = {auc(recalls, precisions):0.3f}$"
    else:
        text1 = f"$AUPRC_{{train}} = {auc(recalls, precisions):0.3f}$"
        
        plt.plot(recalls_test, precisions_test, 'r-', linewidth=2)
        text2 = f"$AUPRC_{{test}} = {auc(recalls_test, precisions_test):0.3f}$"
        plt.gca().text(0.05, 0.13, text2, transform=plt.gca().transAxes, fontsize=13, c='red')
    
    plt.gca().text(0.05, 0.05, text1, transform=plt.gca().transAxes, fontsize=13, c='blue')
    
    if title:
        plt.title(title)
    
    if savetitle:
        fignow.savefig(savetitle, format='png', dpi=300, bbox_inches='tight',
                       transparent=False)
    else:
        plt.show()


def plot_histogram(
    Y, xlabel="Property value", ylabel="No. of compounds", title='', savetitle=None
):
    n, bins, patches = plt.hist(x=Y, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 5)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        if len(title) == 0:
            title = f"{len(Y)} compounds"
        plt.title(title)
    if savetitle:
        plt.savefig(savetitle, format='png', dpi=300, bbox_inches='tight',
                    transparent=False)
    else:
        plt.show()
    plt.show()
