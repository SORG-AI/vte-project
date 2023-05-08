# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:46:47 2022

@author: Bardiya Ak
"""

# Clear workspace variables
from IPython import get_ipython
get_ipython().magic('reset -sf')


# Now that we selected are imputation technique
# evaluate iterative imputation and random forest for the horse colic dataset
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics, tree #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import brier_score_loss
#% define modeling pipeline
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn import metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import util_project as uprj
import util_stats as ustats


seed_val = 42

np.random.seed(seed_val)


#%% Load the cleaned raw dataset
import VTE_PathReader as prj

removed_features = ''#'med_aspirin 81 mg, med_aspirin 325 mg' #'None'
Subset_Or_Full   = 'OnlyOnProphylaxis'#'OnlyOnProphylaxis'#'Subset'


if Subset_Or_Full == 'Subset':
    path_imputed = prj.path_data_imputed_ForModeling_WithoutProphlyaxis
    
elif Subset_Or_Full == 'OnlyOnProphylaxis':
    path_imputed     = prj.path_data_imputed_ForModeling_OnProphlyaxis  
    
elif Subset_Or_Full == 'Full':
    path_imputed = prj.path_data_imputed_ForModeling
    
    
    
with open(prj.path_data_imputed, 'rb') as f:
    Data_imputed = pickle.load(f)
    Data_imputed.drop('MRN/EMPI', axis=1, inplace=True)
    
    
# Reading the variable names
with open(prj.path_my_vars, 'rb') as f:
    all_cat_vars = pickle.load(f)
    
    hp_vars = pickle.load(f)

    predictor_vars = pickle.load(f)
    predictor_cont_vars = pickle.load(f)
    predictor_cat_vars = pickle.load(f)
    
    outcome_vars = pickle.load(f)
    outcome_cont_vars = pickle.load(f)
    outcome_cat_vars = pickle.load(f)
    
# read category names and ticks
cat_dic, cat_len = uprj.get_cat_data(prj.path_description, all_cat_vars, 'ankle_fracture')
tick_list = list(cat_dic.items())

outcomes_varnames = cat_dic[outcome_cat_vars].split(', ')
Target_Name = outcome_vars[0]


#print('Reading the imputed dataframe...')
with open(path_imputed, 'rb') as f:
    test_df = pickle.load(f)
    val_df  = pickle.load(f)
    train_df =  pickle.load(f) 

# a = [key for key in train_df.keys()]
if removed_features == 'None' or removed_features == '':
    pop_list = []
    
else:
    removed_features.split(', ')
    pop_list = removed_features.split(', ') #['Previous_PE','Previous_DVT']
    
# remove the items
for pop_item in pop_list:
    test_df.pop(pop_item)
    val_df.pop(pop_item)
    train_df.pop(pop_item)


# key_vte_lab = [key for key in df_train.columns if '_vte_' in key]
# for key_vte in key_vte_lab:
#     df_train_imputed.drop(key_vte,  axis=1, inplace=True)    
#     df_train.drop(key_vte,  axis=1, inplace=True)
#     df_test.drop(key_vte,  axis=1, inplace=True)


 
 #%% Normalizing and transforming if skewed for continuous variables
cont_var = ['Age','BMI','Hospital_stay','CCI','trauma_to_treatment_days']

FIX_SKEW = True
# Quantile Transform usually performs best
from sklearn.preprocessing import QuantileTransformer

if FIX_SKEW:
    for var in cont_var:    
        q_transform = QuantileTransformer(output_distribution='normal',random_state=seed_val)
        
        train_array          = np.array(train_df[var]).reshape(-1, 1)
        val_array            = np.array(val_df[var]).reshape(-1, 1)
        test_array           = np.array(test_df[var]).reshape(-1, 1)
        
        skew_fix_by_train_df = q_transform.fit_transform(train_array)
        train_df[var] = pd.DataFrame(skew_fix_by_train_df, columns=[var]).squeeze()

        val_df[var] = pd.DataFrame(q_transform.transform(val_array), columns=[var]).squeeze()
        test_df[var] = pd.DataFrame(q_transform.transform(test_array), columns=[var]).squeeze()
        
        del(q_transform)


#%%
for var in cont_var:    
    temp_df = train_df[var].copy(deep=True) # use the train-val set to normalize
    
    train_df[var]   = (train_df[var] - temp_df.mean()) / temp_df.std()
    val_df[var]     = (val_df[var]   - temp_df.mean()) / temp_df.std()
    test_df[var]    = (test_df[var]  - temp_df.mean()) / temp_df.std()
     
    
#%% Reading Training/Test/Validation Sets as Arrays
X_train = np.array(train_df.iloc[:,:-1])
X_val   = np.array(val_df.iloc[:,:-1])
X_test = np.array(test_df.iloc[:,:-1])


y_train = np.array(train_df[Target_Name])
y_val   = np.array(val_df[Target_Name])
y_test  = np.array(test_df[Target_Name])

 

 

#%% Feature Selection& Logistic Regression
train_val_X = pd.concat([train_df.iloc[:,:-1], val_df.iloc[:,:-1]], ignore_index=True)
train_val_y = pd.Series(np.concatenate([y_train, y_val]))


model_name = 'Backward' #'Backward' # 'Chi' or 'Backward'


if model_name == 'Backward':
    included = ustats.backward_regression(train_val_X, train_val_y, threshold_out = 0.1)
    
elif model_name == 'Backward_2':
    included = ustats.backward_regression(train_val_X, train_val_y, threshold_out = 0.1)
    
elif model_name == 'ChiTree':
    sig_corr_chi, ORs, CrossTabs = ustats.run_ChiSquared(Data_imputed, Target_Name, print_on = True, alpha = 0.1)
    
    included = [key for key in sig_corr_chi.index.values if key !=Target_Name and key!='Age'and key!='CCI']
    
elif model_name == 'Matched':
    
    included = ['MVA', 'lab_k', 'cm_none', 'Treatment_of_fracture', 'Smoking',
           'prphlyx_warfarin', 'med_statin', 'cm_cataract', 'VTE_Prophylaxis',
           'prphlyx_aspirin 325 mg', 'Wound_type',
           'gd_hypercoagulable state', 'prphlyx_heparin', 'lab_rbc',
           'cm_arthritis', 'med_enoxaparin', 'med_lisinopril',
           'med_levothyroxine', 'Age', 'Gender', 'BMI']
    
elif model_name == 'Statin_Project':
    
     included = ['Age', 'Gender', 'BMI', 'med_statin']
    
    
    
print(', '.join([a for a in included]))


X_train_mod = np.array(train_val_X[included])
X_test_mod  = np.array(test_df[included])


 
#%% Sampling the Dataset
import util_data as udata

sampling_strategy = 'random undersample'
X_train_under, y_train_under = udata.sample_data(sampling_strategy, X_train_mod, train_val_y, seed_val)



#%%
if model_name == 'Backward':
    model = LogisticRegression(random_state=seed_val)
    
elif model_name == 'Backward_2':
    model = DecisionTreeClassifier(random_state=seed_val, max_depth = 3)
        
elif model_name == 'ChiTree':
    model = DecisionTreeClassifier(random_state=seed_val, max_depth = 5)
else:
    model = LogisticRegression(random_state=seed_val)

# fit a model
# model = LogisticRegression()
model.fit(X_train_under, y_train_under)

y_pred = model.predict(X_test_mod)
yhat = model.predict_proba(X_test_mod)


# retrieve the probabilities for the positive class
yhat_positive = yhat[:, 1]

# calculate the no skill line as the proportion of the positive class
no_skill = len(y_test[y_test==1]) / len(y_test)


_, model_results = ustats.calculate_metrics(y_test, y_pred, yhat_positive, visualize=True)
model_results = pd.Series(model_results).round(3)
print(model_results)

# #### METRICS: calculate inputs for the roc curve
# fpr, tpr, thresholds = roc_curve(y_test, yhat_positive)
# # calculate inputs for the PR curve
precision, recall, thresholds = precision_recall_curve(y_test, yhat_positive)



# # ROC Curve
# plt.figure()
# # plot no skill roc curve - the diagonal line
# plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# # plot roc curve
# plt.plot(fpr, tpr, marker='.', label=model_name)
# # axis labels
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # show the legend
# plt.legend()
# plt.show()
# # calculate and print AUROC
# roc_auc = roc_auc_score(y_test, yhat_positive)
# print('AUROC (C-Index for Logistic Regression): %.3f' % roc_auc)


# ############# PRECISION-RECALL
# plt.figure(dpi=300)
# # plot the no skill precision-recall curve
# plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# # plot PR curve
# plt.plot(recall, precision, marker='.', label=model_name)
# # axis labels
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# # show the legend
# plt.legend()
# plt.show()

# calculate and print PR AUC
# auc_pr = auc(recall, precision)
# print('AUPRC: %.3f' % auc_pr)

# #import the required library
# from numpy import argmax
# # Calculate F-Scores and find the index of ideal score
# fscore = (2 * precision * recall) / (precision + recall)
# ix = argmax(fscore)
# best_thresh = thresholds[ix]
# print('Best Threshold: %f' % (best_thresh))





if model_name == 'ChiTree' or model_name == 'Backward_2':

    fig = plt.figure(figsize=(15,8), dpi= 200)
    out = tree.plot_tree(model, 
                       feature_names=included,  
                       class_names = outcomes_varnames, fontsize= 16,
                       filled=True, label='none', impurity=False)
    for o in out:
        arrow = o.arrow_patch
        if arrow is not None:
            arrow.set_edgecolor('k')
            arrow.set_linewidth(2)



#%%
print("\n--------- Undersampling and Finding the best model ---------")
import util_models

best_model = util_models.find_best_model(X_train_under, y_train_under, seed_val)



#%% RUN THIS AFTER YOU KNOW THE BEST MODEL
# best_model = LogisticRegression(random_state=seed_val)
# best_model = RandomForestClassifier(max_depth=5, n_estimators=5, random_state=seed_val)
# best_model = RandomForestClassifier(max_depth=20, n_estimators=20, random_state=seed_val)
# best_model = MultinomialNB(alpha= 0.9, fit_prior= False)
# best_model =  KNeighborsClassifier(n_neighbors= 2)

# best_model = GaussianNB()
best_model.fit(X_train_under, y_train_under)

# y_pred_train = best_model.predict(X_train_under)
# acc_train = accuracy_score(y_train_under, y_pred_train)
# cm_train  = confusion_matrix(y_train_under, y_pred_train)
# print("\n\nTraining Set:\nClassification Accuracy: %{}".format(acc_train*100))
# print("Confusion Matrix:")
# print(cm_train)
# print(classification_report(y_train, y_pred_train))


y_test_pred_probs = best_model.predict_proba(X_test_mod)
y_test_pred = best_model.predict(X_test_mod)
# acc_test = accuracy_score(y_test, y_test_pred)
# cm_test  = confusion_matrix(y_test, y_test_pred)
# print("\n\nTest Set:\nClassification Accuracy: %{}".format(acc_test*100))
# print("Confusion Matrix:")
# print(cm_test)


_, model_results_nn = ustats.calculate_metrics(y_test, y_test_pred, y_test_pred_probs[:,1], visualize=True)
model_results_nn = pd.Series(model_results_nn).round(3)
print(model_results_nn)

# # calculate the fpr and tpr for all thresholds of the classification
#
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)

# plt.figure(dpi = 600)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# # clf_ori = clf.fit(X_train, y_train)
# # y_pred_test = clf.predict(X_test)
# # acc = accuracy_score(y_test, y_pred_test)
# # cm  = confusion_matrix(y_test, y_pred_test)

# # print("\n\nFull Dataset:")
# # print("Accuracy on the Test Set: %{}".format(acc*100))
# # print("Confusion Matrix:\n", cm)


# # confusion_matrix = ustats.calculate_metrics(y, y_pred)
# tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_test).ravel()

# print('AUROC (C-Index for Logistic Regression): %.3f' % roc_auc)
# auc_pr = auc(recall, precision)
# precision, recall, thresholds = precision_recall_curve(y_test, preds)

# print('AUPRC: %.3f' % auc_pr)
# print('Brier Score: %.3f' % (brier_score_loss(y_test, preds)))

# CM_ForPaper = pd.DataFrame({'Predicted No VTE': [tn, fn], 'Predicted VTE': [fp, tp]}, index=['Observed No VTE', 'Observed VTE'])
# print(CM_ForPaper)
# print('Specificity: {:.2f}'.format(tn/(tn+fp)))
# print('Sensitivity: {:.2f}'.format(tp/(fn+tp)))



# print(classification_report(y_test, y_pred_test))

#%%
df_test = pd.DataFrame(X_test_mod, columns=included)

import shap


explainer = shap.TreeExplainer(best_model)
shap_values = explainer(X_test_mod)
# shap.summary_plot(shap_values, df_test, plot_type="bar")

shap.summary_plot(shap_values, df_test)


# shap.plots.bar(shap_values)

# shap.plots.beeswarm(shap_values)

included_df = pd.Series(included)
sorted_idx = best_model.feature_importances_.argsort()
plt.figure(figsize=(10,10),dpi = 300)
plt.barh(included_df[sorted_idx], best_model.feature_importances_[sorted_idx])
plt.xlabel("Best Model's Feature Importance")



















