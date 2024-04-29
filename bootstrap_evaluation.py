#!/usr/bin/env python
# coding: utf-8


import numpy as np
from sklearn import metrics
from sklearn.utils import resample
from functools import wraps

def calc_conf_mat(y_true, pred_score, threshold, class_name):
    """calculate confusion matrix

    Args:
        y_true (array): true label
        pred_score (array): prediction score
        threshold (float): threshold to convert score to class
        class_name (list): class name to use. Second element should be positive class.
    return:
        tn, fp, fn, tp (int): confusion matrix
    """
    # convert y_pred to class
    y_pred = np.array(list(map(lambda x: 1 if x > threshold else 0, pred_score)))
    y_pred = np.array(list(map(lambda x: class_name[int(x)], y_pred)))
    # y_true = np.array(y_true)

    # create confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, labels=class_name)
    tn, fp, fn, tp = cm.ravel()

    return (tn, fp, fn, tp)


def bootstrap_decorator(metric_func):
    @wraps(metric_func)
    def wrapper(
        y_true,
        pred_score,
        *args,
        bootstrap=False,
        n_boot=1000,
        alpha=0.05,
        random_state=True,
        stratify=False,
        **kwargs,
    ):
        metric = metric_func(y_true, pred_score, *args, **kwargs)
        if bootstrap:
            metric_boot = []
            for i in range(n_boot):
                rs = i if random_state else None
                y_true_boot, pred_score_boot = resample(
                    y_true,
                    pred_score,
                    replace=True,
                    random_state=rs,
                    stratify=y_true if stratify else None,
                )
                metric_boot.append(metric_func(y_true_boot, pred_score_boot, *args, **kwargs))
            metric_boot = np.array(metric_boot)
            ci = np.quantile(metric_boot, [alpha / 2, 1 - alpha / 2])
            return (metric, ci[0], ci[1])
        else:
            return metric

    return wrapper


@bootstrap_decorator
def calc_sen(y_true, pred_score, threshold, class_name):
    """calculate sensitivity

    Args:
        y_true (array): True class
        pred_score (array): prediction score
        threshold (float): threshold to convert score to class
        class_name (list): class name to use. Second element should be positive class.
        bootstrap (bool, optional): True if CI calculation needed. Defaults to False.
        n_boot (int, optional): number of bootstrap needed (ignored when bootstrap == False). Defaults to 1000.
        alpha (float, optional): CI level. Defaults to 0.05.
        random_state (bool, optional): random state for bootstrap. Defaults to True.
        stratify (bool, optional): stratify for bootstrap. Defaults to False.
    return:
        sen (float): sensitivity
        ci_lower (float): left confidence interval (if booststrap == True)
        ci_upper (float): right confidence interval (if booststrap == True)
    """
    tn, fp, fn, tp = calc_conf_mat(y_true, pred_score, threshold, class_name)
    if tp + fn == 0:
        return np.nan
    else:
        return tp / (tp + fn)

    
    
    
@bootstrap_decorator
def calc_acc(y_true, pred_score, acc_threshold, class_name):
    """calculate sensitivity

    Args:
        y_true (array): True class
        pred_score (array): prediction score
        threshold (float): threshold to convert score to class
        class_name (list): class name to use. Second element should be positive class.
        bootstrap (bool, optional): True if CI calculation needed. Defaults to False.
        n_boot (int, optional): number of bootstrap needed (ignored when bootstrap == False). Defaults to 1000.
        alpha (float, optional): CI level. Defaults to 0.05.
        random_state (bool, optional): random state for bootstrap. Defaults to True.
        stratify (bool, optional): stratify for bootstrap. Defaults to False.
    return:
        sen (float): sensitivity
        ci_lower (float): left confidence interval (if booststrap == True)
        ci_upper (float): right confidence interval (if booststrap == True)
    """       
    tn, fp, fn, tp = calc_conf_mat(y_true, pred_score, acc_threshold, class_name)
    if tp + fn == 0:
        return np.nan
    else:
        return (tp + tn) / (tp + fn + tn + fp)
    
    


@bootstrap_decorator
def calc_spe(y_true, pred_score, threshold, class_name):
    """calculate specificity

    Args:
        y_true (array): True class
        pred_score (array): prediction score
        threshold (float): threshold to convert score to class
        class_name (list): class name to use. Second element should be positive class.
        bootstrap (bool, optional): True if CI calculation needed. Defaults to False.
        n_boot (int, optional): number of bootstrap needed (ignored when bootstrap == False). Defaults to 1000.
        alpha (float, optional): CI level. Defaults to 0.05.
        random_state (int, optional): random state for bootstrap. Defaults to 0.
        stratify (bool, optional): stratify for bootstrap. Defaults to False.
    return:
        sen (float): sensitivity
        ci_lower (float): left confidence interval (if booststrap == True)
        ci_upper (float): right confidence interval (if booststrap == True)
    """
    tn, fp, fn, tp = calc_conf_mat(y_true, pred_score, threshold, class_name)
    if tn + fp == 0:
        return np.nan
    else:
        return tn / (tn + fp)


@bootstrap_decorator
def calc_auc(y_true, pred_score, pos_label):
    """calculate AUC

    Args:
        y_true (array): True class
        pred_score (array): prediction score
        pos_label (str): positive class name
        bootstrap (bool, optional): True if CI calculation needed. Defaults to False.
        n_boot (int, optional): number of bootstrap needed (ignored when bootstrap == False). Defaults to 1000.
        alpha (float, optional): CI level. Defaults to 0.05.
        random_state (int, optional): random state for bootstrap. Defaults to 0.
        stratify (bool, optional): stratify for bootstrap. Defaults to False.
    return:
        auc (float): AUC
        ci_lower (float): left confidence interval (if booststrap == True)
        ci_upper (float): right confidence interval (if booststrap == True)
    """
    if pos_label in np.unique(y_true):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, pred_score, pos_label=pos_label)
        return metrics.auc(fpr, tpr)
    else:
        return np.nan



import re
import glob
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from collections import OrderedDict
from PIL import Image
from skimage.color import rgb2gray
from tensorflow.python.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score
from plotnine import *



def annot_metric(df, metric_name, metric_ci):
    annot = []
    for i in df.index:
        metric = df[metric_name][i] * 100
        ci_left = df[metric_ci][i][0] * 100
        ci_right = df[metric_ci][i][1] * 100
        annot.append("{0}% ({1}% to {2}%)".format(np.round(metric, 1), np.round(ci_left, 1), np.round(ci_right, 1)))
    return annot



def calculate_metric(df, y_true, pred_score, threshold, acc_threshold, class_name, pos_label, bootstrap_n):
    sen_list = []
    sen_ci_list = []
    auc_list = []
    auc_ci_list = []
    acc_list = []
    acc_ci_list = []
    ppv_list = []
    ppv_ci_list = []
    n_normal = []
    n_cancer = []
    individual_detected = []
    individual_analyzed = []

    ## Sensitivity
    try:
        sen, sen_ci_lower, sen_ci_upper = calc_sen(y_true, pred_score, threshold, class_name, bootstrap=True, n_boot=bootstrap_n,stratify=True)
        sen_list.append(sen)
        sen_ci_list.append((np.round(sen_ci_lower, 3), np.round(sen_ci_upper, 3)))

    except ZeroDivisionError:
        sen_list.append("N/A")
        sen_ci_list.append("N/A")

    ## AUC
    try:
        auc, auc_ci_lower, auc_ci_upper = calc_auc(y_true, pred_score, pos_label, bootstrap=True, n_boot=bootstrap_n,stratify=True)
        auc_list.append(auc)
        auc_ci_list.append((np.round(auc_ci_lower, 3), np.round(auc_ci_upper, 3)))

    except ZeroDivisionError:
        auc_list.append("N/A")
        auc_ci_list.append("N/A")
        
    ## Accuracy
    try:
        acc, acc_ci_lower, acc_ci_upper = calc_acc(y_true, pred_score, acc_threshold, class_name, bootstrap=True, n_boot=bootstrap_n,stratify=True)
        acc_list.append(acc)
        acc_ci_list.append((np.round(acc_ci_lower, 3), np.round(acc_ci_upper, 3)))

    except ZeroDivisionError:
        acc_list.append("N/A")
        acc_ci_list.append("N/A")


    n_normal.append(df.loc[df.true_class == "Normal", :].shape[0])
    n_cancer.append(df.loc[df.true_class != "Normal", :].shape[0])
    individual_analyzed.append(df.loc[df.true_class != "Normal", :].shape[0])

    try:
        results = pd.DataFrame({"num_normal": n_normal,
                                "num_cancer": n_cancer,
                                "individual_analyzed": individual_analyzed,
                                "sensitivity": sen_list,
                                "sensitivity_95_ci": sen_ci_list,
                                "Accuracy": acc_list,
                                "Accuracy_95_ci": acc_ci_list,
                                "AUC": auc_list,
                                "AUC_95_ci": auc_ci_list})
    except:
        results = pd.DataFrame({"num_normal": n_normal,
                                "num_cancer": n_cancer,
                                "individual_analyzed": individual_analyzed,
                                "sensitivity": sen_list,
                                "sensitivity_95_ci": sen_ci_list,
                                "Accuracy": acc_list,
                                "Accuracy_95_ci": acc_ci_list,
                                "AUC": auc_list,
                                "AUC_95_ci": auc_ci_list}, index=[0])

    results.loc[:, "specificity_cutoff"] = specificity_cutoff


    annot_list = ["sen_annot", "acc_annot", "auc_annot"]
    metric_list = ["sensitivity", "Accuracy", "AUC"]
    ci_list = ["sensitivity_95_ci", "Accuracy_95_ci", "AUC_95_ci"]
    
    for c, annot in enumerate(annot_list):
        results.loc[:, annot] = annot_metric(results, metric_list[c], ci_list[c])

    return results



def plot_sensitivity(df, x_col, cutoff=95, data_type=False, ax=None):
    df = df.loc[df.specificity_cutoff == cutoff, :].copy()

    x_val = list(map(lambda x: "{0} \n ({1}/{2})".format(df[x_col][x], str(df["individual_detected"][x]), str(df["individual_analyzed"][x])), df.index))
    text_labels = df.sensitivity.to_list()
    err = []
    for i in df.index:
        lower = df.loc[i, "sensitivity"] -  df.loc[i, "sensitivity_95_ci"][0]
        upper = df.loc[i, "sensitivity_95_ci"][1] - df.loc[i, "sensitivity"]
        err.append((lower, upper))
    err = np.transpose(np.array(err))

    if ax:
        ax.errorbar(x_val, df.sensitivity, yerr=err, fmt='o', color="navy", capsize=5)
        ax.scatter(x_val, df.sensitivity, s=50, color="navy")
        ax.grid(axis="y")
        for tick, height in zip(range(len(text_labels)), df.sensitivity_95_ci.to_list()):
            ax.text(tick, height[1] + 0.02, "{0}%".format(np.round(text_labels[tick] * 100, 1)), horizontalalignment="center", color="black")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(-0.3, len(text_labels) -1 + 0.3)
        ax.set_ylabel("Sensitivity (95% CI)", fontweight="bold")
        ax.set_yticks(np.arange(0,1.1, 0.25))
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    
    else:
        plt.errorbar(x_val, df.sensitivity, yerr=err, fmt='ko', capsize=5)




####################################################
InputFile = "input.txt"
OutputPrefix = "output"
NUM_BOOTSTRAP = 2000
cutoff_values = [95, 98]
####################################################



df = pd.read_csv(InputFile, sep="\t")
df.head()


## DataType
metric_dfs = []

for specificity_cutoff in cutoff_values:
    float_cuoff = np.percentile(df.loc[ (df.true_class == "Normal"), "pred_prob"], specificity_cutoff)  
    correct = df.loc[(df.pred_prob >= float_cuoff) & (df.true_class != "Normal"), :].shape[0]
    metric_df = calculate_metric(df,df.true_class.values, df.pred_prob.values, 
                                     float_cuoff, 0.5,["Normal","Cancer"],"Cancer",bootstrap_n=NUM_BOOTSTRAP)

    
    metric_df.loc[:, "individual_detected"] = correct
    metric_dfs.append(metric_df)
    del metric_df
data_type_df = pd.concat(metric_dfs)
data_type_df.reset_index(drop=True, inplace=True)
data_type_df.to_csv("{}_data_type_metric.csv".format(OutputPrefix), index=False)
data_type_df.head()



### Stage
metric_dfs = []

for specificity_cutoff in cutoff_values:
    float_cuoff = np.percentile(df.loc[ (df.true_class == "Normal"), "pred_prob"], specificity_cutoff) 
    for stage in set(df.loc[~df.stage.isin(["Normal"]), :].stage.to_list()):
        correct = df.loc[(df.stage == stage) & (df.pred_prob >= float_cuoff) & (df.true_class != "Normal"), :].shape[0]
        metric_df = calculate_metric(df.loc[(df.stage == stage) | (df.stage == "Normal"), :],
                                     df.loc[(df.stage == stage) | (df.stage == "Normal"), :].true_class.values, 
                                     df.loc[(df.stage == stage) | (df.stage == "Normal"), :].pred_prob.values, 
                                     float_cuoff, 0.5,["Normal","Cancer"],"Cancer",bootstrap_n=NUM_BOOTSTRAP)
        
        metric_df.loc[:, "individual_detected"] = correct
        metric_df.loc[:, "stage"] = stage
        metric_dfs.append(metric_df)
        del metric_df
stage_metric_df = pd.concat(metric_dfs)
stage_metric_df.sort_values(["stage"], inplace=True)
stage_metric_df.reset_index(drop=True, inplace=True)
stage_metric_df.to_csv("{}_stage_metric.csv".format(OutputPrefix), index=False)
stage_metric_df.head()



### Subtype
metric_dfs = []

for specificity_cutoff in cutoff_values:
    float_cuoff = np.percentile(df.loc[ (df.true_class == "Normal"), "pred_prob"], specificity_cutoff) 
    for cancer_type in set(df.loc[~df.subtype.isin(["Normal"]), :].subtype.to_list()):
        correct = df.loc[(df.subtype == cancer_type) & (df.pred_prob >= float_cuoff) & (df.true_class != "Normal"), :].shape[0]
        metric_df = calculate_metric(df.loc[(df.subtype == cancer_type) | (df.subtype == "Normal"), :],
                                     df.loc[(df.subtype == cancer_type) | (df.subtype == "Normal"), :].true_class.values, 
                                     df.loc[(df.subtype == cancer_type) | (df.subtype == "Normal"), :].pred_prob.values, 
                                     float_cuoff, 0.5,["Normal","Cancer"],"Cancer",bootstrap_n=NUM_BOOTSTRAP)

        metric_df.loc[:, "individual_detected"] = correct
        metric_df.loc[:, "cancer_type"] = cancer_type
        metric_dfs.append(metric_df)
        del metric_df
subtype_metric_df = pd.concat(metric_dfs)
subtype_metric_df.sort_values(["cancer_type"], inplace=True)
subtype_metric_df.reset_index(drop=True, inplace=True)
subtype_metric_df.to_csv("{}_subtype_metric.csv".format(OutputPrefix), index=False)
subtype_metric_df.head()


import matplotlib
matplotlib.use('agg')

plt.rcdefaults()

fig, axes = plt.subplots(2, 2, figsize=(14,10),facecolor='white')

plot_sensitivity(stage_metric_df, "stage", cutoff=95, ax=axes[0,0])
axes[0,0].set_xlabel("Stage", fontweight="bold")
axes[0,0].set_title("95% specificity", fontweight="bold")
plot_sensitivity(stage_metric_df, "stage", cutoff=98, ax=axes[0,1])
axes[0,1].set_xlabel("Stage", fontweight="bold")
axes[0,1].set_title("98% specificity", fontweight="bold")

plot_sensitivity(subtype_metric_df, "cancer_type", cutoff=95, ax=axes[1,0])
axes[1,0].set_xlabel("Subtype", fontweight="bold")
axes[1,0].set_title("95% specificity", fontweight="bold")
plot_sensitivity(subtype_metric_df, "cancer_type", cutoff=98, ax=axes[1,1])
axes[1,1].set_xlabel("Subtype", fontweight="bold")
axes[1,1].set_title("98% specificity", fontweight="bold")


for i, label in enumerate(["a", "b"]):
    axes[i, 0].text(-0.2, 0.5, f"{label}", transform=axes[i, 0].transAxes, ha='center', va='center', fontweight='bold', fontsize=30)

plt.subplots_adjust(hspace=0.3, left=0.1, right=0.95, bottom=0.1, top=0.95)
plt.savefig("Figure_4.tiff",dpi=300)



