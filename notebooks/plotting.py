# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:23:53 2021

Contains plotting functions for MACAW project.

@author: Vincent
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import auc

from matplotlib import rc
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['figure.dpi'] = 96
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rc('figure', figsize=(5.0, 4.0))

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
    plt.figure(figsize=(4.4, 4.0))
    
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
    plt.errorbar(x=x, y=y, yerr=y_train_std, color='blue', fmt='.', elinewidth=.5, alpha=0.5)

    if x_test is not None:
        plt.errorbar(x=x_test, y=y_test, yerr=y_test_std, color='red', fmt='.', elinewidth=.5, alpha=0.5)
        text1 = f"$R^2_{{train}} = {r2_score(x,y):0.2f}$"
        text2 = f"$R^2_{{test}} = {r2_score(x_test,y_test):0.2f}$"

        plt.gca().text(0.05, 0.93, text1, transform=plt.gca().transAxes, fontsize=9., c='blue')
        plt.gca().text(0.05, 0.85, text2, transform=plt.gca().transAxes, fontsize=9., c='red')

    else:
        text1 = f"$R^2= {r2_score(x,y):0.2f}$"
        plt.gca().text(0.05, 0.93, text1, transform=plt.gca().transAxes, fontsize=9.)

    # Title and labels
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save the figure into 300 dpi
    if savetitle:
        plt.savefig(savetitle, format='svg', dpi=300, bbox_inches='tight', transparent=False)
    else:
        plt.show()


def plot_precision_vs_recall(precisions, recalls, precisions_test=None, recalls_test=None, title=None, savetitle=None):
    
    plt.figure(figsize=(4.4, 4.0))
    plt.plot(recalls, precisions, 'b-', linewidth=1.)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])

    if precisions_test is None:
        text1 = f"$AUPRC = {auc(recalls, precisions):0.3f}$"
    else:
        text1 = f"$AUPRC_{{train}} = {auc(recalls, precisions):0.3f}$"
        
        plt.plot(recalls_test, precisions_test, 'r-', linewidth=2)
        text2 = f"$AUPRC_{{test}} = {auc(recalls_test, precisions_test):0.3f}$"
        plt.gca().text(0.05, 0.13, text2, transform=plt.gca().transAxes, fontsize=9., c='red')
    
    plt.gca().text(0.05, 0.05, text1, transform=plt.gca().transAxes, fontsize=9., c='blue')
    
    if title:
        plt.title(title)
    
    if savetitle:
        plt.savefig(savetitle, format='svg', dpi=300, bbox_inches='tight', transparent=False)
    else:
        plt.show()


def plot_histogram(
    Y, xlabel="Property value", ylabel="No. of compounds", title='', savetitle=None
):
    plt.figure(figsize=(5.0, 3.5))
    n, bins, patches = plt.hist(x=Y, bins=20, alpha=1., rwidth=1., edgecolor='black', linewidth=.5)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        if len(title) == 0:
            title = f"{len(Y)} compounds"
        plt.title(title)
    if savetitle:
        plt.savefig(savetitle, format='svg', dpi=300, bbox_inches='tight', transparent=False)
    else:
        plt.show()
