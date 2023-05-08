# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:32:06 2021

@author: Bardiya
"""

import numpy as np
import pandas as pd
from scipy import stats # statistics toolbox
from itertools import chain, compress, combinations # to filter lists in one line
import matplotlib.pyplot as plt
from collections import defaultdict



#%% Statistical Analysis (Pearson's R, Chi-Squared, One-Way ANOVA)
#%%% Pearson's R
def run_PerasonTest(dataframe, outcome_var, print_on = True, alpha = 0.05):
    m = dataframe.shape[1] # number of features

    PR_r_vals = np.empty((m,1))
    PR_p_vals = np.empty((m,1))

    for iCol, key in enumerate(dataframe.keys()):
        ### Pearson R
        PR_r_vals[iCol], PR_p_vals[iCol] = stats.pearsonr(dataframe[outcome_var], dataframe[key])

    crit = PR_p_vals < alpha
    sig_PR_names = list(compress(dataframe.keys(), crit))
    sig_PR_r = list(compress(PR_r_vals, crit))
    sig_PR_p = list(compress(PR_p_vals, crit))
    
    sig_corr_pearson = pd.DataFrame(zip(sig_PR_r,sig_PR_p), index=sig_PR_names, columns=['R-Adj','p-val'])
    sig_corr_pearson = sig_corr_pearson.sort_values(by='p-val')
    
    if print_on:
        print('\n\nUsing Persons R:')    
        print(sig_corr_pearson.sort_values(by='p-val').to_markdown())
    
    
    
    return sig_corr_pearson
    
    
#%%% Chi-Squared Test


def run_ChiSquared(dataframe, outcome_var, print_on = True, alpha = 0.05, sort_on = False):
    m = dataframe.shape[1] # number of features

    chi2 = np.empty((m,1))
    p_value = np.empty((m,1))
    
    for iCol in range(m):
        crosstab = pd.crosstab(dataframe.iloc[:,iCol], dataframe[outcome_var], margins=True)    
        
        chi2[iCol], p_value[iCol], _, _ = stats.chi2_contingency(crosstab)
        
       
    crit = p_value < alpha
    sig_names = list(compress(dataframe.keys(), crit))
    sig_r = list(compress(chi2, crit))
    sig_p = list(compress(p_value, crit))
    
    sig_corr_chi = pd.DataFrame(zip(sig_r,sig_p), index=sig_names, columns=['Chi Value','p-val'])
    
    OR = defaultdict()
    CI_dict = defaultdict()
    OR_CI = defaultdict()
    p_vals = defaultdict()
    CrossTabs = defaultdict()
    for i in sig_names:
        crosstab = pd.crosstab(dataframe[i], dataframe[outcome_var], margins=True)
        crosstab_np = crosstab.to_numpy()
        cross_tab_values = crosstab_np[:2,:2]
        
        if cross_tab_values.min() >= 0: # if we have less than 10 in a group, this doesn't mean much
            if np.all(crosstab[:2][:2]):
                OR_temp = (crosstab[0][0]/crosstab[0][1])/ (crosstab[1][0]/crosstab[1][1])
            
            # CI = 1.96*np.sqrt((1/crosstab[0][0]) + (1/crosstab[0][1]) + (1/crosstab[1][0]) + (1/crosstab[1][1]))
            
                
                CI_temp = 1.96*np.sqrt((1/crosstab[0][0]) + (1/crosstab[0][1]) + (1/crosstab[1][0]) + (1/crosstab[1][1]))
    
                CI_upper = np.exp(np.log(OR_temp) + CI_temp)
                CI_lower = np.exp(np.log(OR_temp) - CI_temp)
                
                
                OR[i]    = '{:.2f}'.format(OR_temp)
                CI_dict[i]    = '{:.2f}'.format(CI_temp)
                OR_CI[i] = '{:.2f} - {:.2f}'.format(CI_lower, CI_upper)
                p_vals[i] = '{:.4f}'.format(sig_corr_chi.loc[i]['p-val'][0])
                
                CrossTabs[i] = crosstab
                
            else:
                continue
            # print('\n')
            # print(crosstab)


    OR_ser = pd.Series(OR, name='OR')
    OR_CI_ser = pd.Series(OR_CI, name='95% CI')
    pval_ser = pd.Series(p_vals, name='p-value')
    CI_ser = pd.Series(CI_dict, name='CI Value')
    OR_df = pd.concat([OR_ser, OR_CI_ser, pval_ser, CI_ser], axis=1)
    
    if print_on:
        print('\nChi-Squared Test:')

        
        print('Odd\'s Ratio:')   
        if sort_on:
            print(OR_df.sort_values(by='p-value').to_markdown())
        else:
            print(OR_df.sort_values(by='OR',ascending=False).to_markdown())

    if sort_on:
        # sig_corr_chi = sig_corr_chi.sort_values(by='p-val')    
        OR_df = OR_df.sort_values(by='p-value')           

        
    return sig_corr_chi, OR_df, CrossTabs


#%%% GLM
import statsmodels.api as sm
import statsmodels.formula.api as glm


def run_GLM(dataframe, TargetName, Features, alpha = 0.05):

    ind = sm.cov_struct.Exchangeable()
    # Fit the model
    # wells_fit = glm(formula = model_formula, data = dataframe, family = model_family)
    
    x_var = dataframe[Features]#['Age', 'Height', 'Weight', 'BMI','HBA1C_Level']]
    x_var = sm.add_constant(x_var)
    glm_fit = sm.GLM(dataframe[TargetName],  x_var).fit() #, family = model_family
    
    results_summary = glm_fit.summary()
    results_as_html = results_summary.tables[1].as_html()
    results_df_temp = pd.read_html(results_as_html, header=0, index_col=0)[0]
    
    results_df = results_df_temp.sort_values(by=['P>|z|'])[['coef','P>|z|']]


    results_sig = results_df.loc[results_df['P>|z|'] < alpha]
    print(results_sig.sort_values(by='P>|z|',ascending=True).to_markdown())
        
    return results_df, glm_fit


#%%% One-Way ANOVA
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
from scipy import stats
from collections import defaultdict
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def run_OneWayANOVA(data_stats, varNames, Target_Name, outcomes_varnames, print_on = True):
    full_table = defaultdict()
    full_table_access = defaultdict()
    
    for varName in varNames:
        # print(varName)
        tukey = pairwise_tukeyhsd(endog=data_stats[varName],
                                  groups=data_stats[Target_Name],
                                  alpha=0.05)
        
        temp_access = [] # this is for the table we need to create so we can access mean and std individually
        temp_tbl = [] # this is mostly for the visualization (+/-) sign will be there
        for case in outcomes_varnames:
            this_set = data_stats.loc[data_stats[Target_Name] == case, varName]
            meanT = np.mean(this_set)
            sdT = np.std(this_set)
            
            temp_access.append(meanT)
            temp_access.append(sdT)
            
            temp_tbl.append('{:.1f} ± {:.1f}'.format(meanT, sdT))
            
            
        temp_tbl.append('{:.1f}'.format(tukey.meandiffs[0]))
        temp_tbl.append('{:.3f}'.format(tukey.pvalues[0]))
        full_table[varName] = temp_tbl
        
        
        temp_access.append(tukey.meandiffs[0])
        temp_access.append(tukey.pvalues[0])
        
        
        full_table_access[varName] = temp_access
        
    # print(tukey.summary())
    
    
    statistics_table = pd.DataFrame(full_table, index = outcomes_varnames+['Mean Difference', 'p-val']).T
    
    if print_on:
        print('\n')
        print(statistics_table)
        print('\n* Mean Difference is {} Cohort subtracted by {} Cohort'.format(outcomes_varnames[1], outcomes_varnames[0]))


    headers = [outcomes_varnames[0]+'_Mean',
               outcomes_varnames[0]+'_SD',
               outcomes_varnames[1]+'_Mean',
               outcomes_varnames[1]+'_SD',
               'Mean Difference',
               'p-val']
    
    access_table = pd.DataFrame(full_table_access, index = headers).T


    return statistics_table, access_table



#%% Backward Regression

def backward_regression(X, y,
                           threshold_out = 0.05, 
                           verbose = True):
    ### To make sure results are consistent
    import random
    import numpy as np
    import statsmodels.api as sm
    np.random.seed(42)
    random.seed(42)
    ###
    
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval= pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax(axis=0, skipna=True)
            included.remove(worst_feature)
            if verbose:
                print('Drop {} with p-value {}'.format(worst_feature, worst_pval))
        if not changed:
            break
        
    return included


#%%
def calculate_metrics(y_true, y_pred, y_pred_prob=[None], visualize=False):
    from sklearn import metrics
    # labels = y_true
    # predictions = y_pred_prob
    y_base = np.zeros((len(y_true), 1)) #[1 if i == 1 else 0 for i in y_true]
    no_skill = len(y_true[y_true==1]) / len(y_true)


    f1score = metrics.f1_score(y_true, y_pred)
    print('----------------------')
    print('F1-Score:\t\t\t\t{:.2f}'.format(f1score))


    print('\nPrediction Accuracy is {:.2f}'.format(metrics.accuracy_score(y_true, y_pred)))
    print('Baseline (0 Recall) is {:.2f}'.format(metrics.accuracy_score(y_true, y_base)))

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()


    f1score_check = 2*tp/(2*tp+fp+fn)
    # try:
    #     assert f1score_check == f1score
    # except:
    #     print('Not Sure Why?')
    #     print(f1score_check)
    #     print(f1score)
    
    f1score = f1score_check
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    
    print('\nPrecision (PPV):\t\t{:.2f}'.format(ppv))

    print('Sensitivity (Recall):\t{:.2f}'.format(sensitivity))
    print('Specificity:\t\t\t{:.2f}'.format(specificity))

    
    if y_pred_prob.any():
        roc_auc = metrics.roc_auc_score(y_true, y_pred_prob)
        print('AUROC (C-Index for Logistic Regression): %.3f' % roc_auc)
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_prob)
        prc_auc = metrics.auc(recall, precision)
        
        if visualize:
            
            plot_roc(fpr, tpr, roc_auc)

            plot_prc(recall, precision, prc_auc, no_skill)
    else:
        roc_auc = np.nan
        prc_auc = np.nan
        
        
    brier = metrics.brier_score_loss(y_true, y_pred)
    print('Brier Score:\t\t\t%.2f' % (brier))
    
    print('\nConfusion Matrix:')
    CM_ForPaper = pd.DataFrame({'Predicted No VTE': [tn, fn], 'Predicted VTE': [fp, tp]}, index=['Observed No VTE', 'Observed VTE'])
    print(CM_ForPaper)
    
    print('\nReport:')
    print(metrics.classification_report(y_true, y_pred))


    model_performance = {
        'Baseline Accuracy - 0 Recall': metrics.accuracy_score(y_true, y_base),
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'AUROC': roc_auc,
        'AUPRC': prc_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Youdens Index': sensitivity+specificity-1,
        'Brier Score': brier,
        'PPV': ppv,
        'NPV': npv,
        'F1 Score': f1score,
        'LR+': sensitivity / (1-specificity) ,
        'LR-': (1-sensitivity) / specificity 
        }

    print('----------------------')
    return CM_ForPaper, model_performance




### Metrics from CM
def calculate_metrics_fromCM(tn, fp, fn, tp):
    
    population = tn + fp + fn + tp
    
    accuracy = (tp+tn)/population
    f1score = 2*tp/(2*tp+fp+fn)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)

    model_performance = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1 Score': f1score,
        'LR+': sensitivity / (1-specificity) ,
        'LR-': (1-sensitivity) / specificity 
        }
    
    return model_performance
    
    
#### Visualization
def plot_roc(fpr, tpr, roc_auc):
    plt.figure(dpi = 600)
    plt.title('Receiver Operating Characteristic')
    
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plot the no skill roc curve
    plt.plot([0, 1], [0, 1],'r--', label='No Skill')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
    
def plot_prc(recall, precision, prc_auc, no_skill):
    plt.figure(dpi = 600)
    plt.title('Precision-Recall Curve')
    
    # plot PR curve
    plt.plot(recall, precision, 'b-.', label = 'AUC = %0.2f' % prc_auc)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], 'r--', label='No Skill')
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc = 'lower right')
    plt.show()
    


#%%
def log_my_model(logfile_path,
                 model_created, model_path,
                 model_name, model_comments, data_comments,
                 model_encoder = False,
                 y_true=[None], y_pred=[None], y_pred_prob=np.array([None]), disease_class=1):
    import os
    import pandas as pd
    from sklearn import metrics
    from datetime import datetime
    cur_date = datetime.now().strftime("%m%d%y_%H%M%S")
    
    overall_acc = metrics.accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1score = metrics.f1_score(y_true, y_pred)
    if y_pred_prob.any():
        roc_auc = metrics.roc_auc_score(y_true, y_pred_prob)
    else:
        roc_auc = np.nan
    brier_score = metrics.brier_score_loss(y_true, y_pred)
    
    # baseline accuracy
    if disease_class == 1:
        y_base = np.zeros((len(y_true), 1))
    else:
        y_base = np.ones((len(y_true), 1))
    baseline_acc = metrics.accuracy_score(y_true, y_base)
    

    # wrap-up report
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    save_path = os.path.join(model_path, model_name+'_'+cur_date+'.h5')

    # saving the model
    try:
        model_created.save(save_path)
        print('--- [MODEL] saved the tensorflow model at {}'.format(save_path))

    except:
        from joblib import dump, load
        dump(model_created, save_path) 
        print('--- [MODEL] saved the sklearn model at {}'.format(save_path))
    
    # save encoder, if exists
    if model_encoder:
        save_path_encoder = os.path.join(model_path, model_name+'_encoder_'+cur_date+'.h5')
        model_encoder.save(save_path_encoder)
        print('--- [ENCODER] saved the tensorflow model at {}'.format(save_path_encoder))       
    else:
        save_path_encoder = 'NAN'
        
    
    log_list = [model_name, cur_date, model_comments, data_comments,
                sensitivity, specificity,
                precision, recall, f1score,
                tn, fp, fn, tp,
                brier_score, roc_auc,
                overall_acc, baseline_acc,
                save_path, save_path_encoder]
        
        
    # save log files    
    header_list = ['model_name', 'cur_date', 'model_comments', 'data_comments',\
                                                'sensitivity', 'specificity',\
                                                'precision', 'recall', 'f1score',\
                                                'tn', 'fp', 'fn', 'tp',\
                                                'brier_score','auroc',\
                                                'overall_acc', 'baseline_acc',\
                                                'save_path',
                                                'save_path_encoder']
    df_log = pd.DataFrame(data=[log_list], columns = header_list)
        
    if os.path.isfile(logfile_path):
        df_log.to_csv(logfile_path, mode='a', index=False, header=False)

    else:
        # else it exists so append
        df_log.to_csv(logfile_path, index=False)
        print("--- [INFO] Log File is Created...")
    
    return log_list

    
    


#%% Bin Values
def bin_column(dataframe, column_name, bin_list = [0, 50, 60, 70, 80, 90, 200], bin_names = None):
    new_name = column_name+'_Bins'
    labels = list(range(len(bin_list)-1))

    dataframe[new_name] = pd.cut(dataframe[column_name].astype(float), bins= bin_list, labels=labels, right = False, include_lowest = True).astype(int)
    
    if bin_names:
        dataframe[new_name] = dataframe[new_name].replace(to_replace=np.sort(dataframe[new_name].unique()), value=bin_names.split(', ')).astype('category')
    
    return dataframe


#%%
# #%%%%
# def barplot_in_percent(data_wCatNames, outcome_varName, cat_dic, cat_name, name_cleaning_idx = 0, rename_dict = None):
#     df_cat = data_wCatNames.loc[:,[cat_name, outcome_varName]]
    
#     outcome_tickNames = cat_dic[outcome_varName].split(', ')
#     print('\nOutcome is {}, classes are: {}'.format(outcome_varName, outcome_tickNames))
    
#     df_outcome_0  = df_cat[outcome_varName] == outcome_tickNames[0]
#     df_outcome_1  = df_cat[outcome_varName] == outcome_tickNames[1]
    
#     cat_tick_vals = cat_dic[cat_name].split(', ')
#     print('--- Variable is {}, classes are: {}'.format(cat_name, cat_tick_vals))

#     #color_list = ['darkblue','lightblue','cyan','red','orange']
    
#     n_ticks = len(cat_tick_vals)
#     tick_count_outcome_0 = np.empty(n_ticks)
#     tick_percent_outcome_0 = np.empty(n_ticks)
        
#     tick_count_outcome_1 = np.empty(n_ticks)
#     tick_percent_outcome_1 = np.empty(n_ticks)
#     # tick_id_list = np.sort(df_cat[cat_name].unique())
#     # print(tick_id_list, cat_tick_vals)
#     for tick_id, tick_name in enumerate(cat_tick_vals):
#         try:
#             this_class = df_cat[cat_name] == cat_tick_vals[tick_id]
            
#             if this_class.sum() == 0:
#                 print('nothing in {} class and variable...'.format(tick_name))
#                 continue

#             this_class_0_count = (this_class & df_outcome_0).sum()
#             tick_percent_outcome_0[tick_id] = 100*this_class_0_count/this_class.sum()
#             tick_count_outcome_0[tick_id] = this_class_0_count

#             this_class_1_count = (this_class & df_outcome_1).sum()
#             tick_percent_outcome_1[tick_id] = 100*this_class_1_count/this_class.sum()
#             tick_count_outcome_1[tick_id] = this_class_1_count
            
#         except:
#             raise ValueError('Something went wrong in this loop. Not sure why?!')
        
        
#     try:
#         plt.bar(cat_tick_vals, tick_percent_outcome_0, label='y1', color=['r'])
        
#         plt.bar(cat_tick_vals, tick_percent_outcome_1,
#                 bottom=tick_percent_outcome_0, label='y2', color=['g'])
#     except:
#         pass
    
#     try:
#         plt.xlabel(rename_dict[cat_name])
#         plt.yticks([])
#     except:
#         plt.xlabel(cat_name[name_cleaning_idx:].capitalize())
#         plt.yticks([])
        
    
#     txt_font_size_list = [22, 22, 20, 18, 16, 12, 12, 10]
#     txt_font_size = txt_font_size_list[n_ticks-1]

    
#     txt_y0 = tick_percent_outcome_0/2
#     txt_y1 = tick_percent_outcome_0 + tick_percent_outcome_1/2
#     for xpos, ypos, percent, count in zip(cat_tick_vals, txt_y0,
#                                 tick_percent_outcome_0,
#                                 tick_count_outcome_0):
#         plt.text(xpos, ypos, "{:.0f}%\n({:.0f})".format(percent, count),
#                   ha="center", va="center", fontsize = txt_font_size, c= 'w', fontweight='bold')
        
#     for xpos, ypos, percent, count in zip(cat_tick_vals, txt_y1,
#                                 tick_percent_outcome_1,
#                                 tick_count_outcome_1):
#         plt.text(xpos, ypos, "{:.0f}%\n({:.0f})".format(percent, count),
#                   ha="center", va="center", fontsize = txt_font_size, c= 'w', fontweight='bold')
        
        
        
        
        
        
#%%%%
def barplot_in_percent_general(data_wCatNames,  xName, yName, ticksDictionary, rename_index = 0,\
                               rename_dict = None, pcnt_control = 20, viz_style = 0,\
                               col_list=[['r'],['g'],['b'],['c'],['m'],['k']], text_color='w', label_fontsize = 12):
    
    #### Data Prep
    df_cat = data_wCatNames.loc[:,[xName, yName]]
    
    outcome_tickNames = ticksDictionary[yName].split(', ')
    nX = len(outcome_tickNames)
    if viz_style==0: print('\nOutcome is {}, classes are: {}'.format(yName, outcome_tickNames))
    
    cat_tick_vals = ticksDictionary[xName].split(', ')
    n_ticks = len(cat_tick_vals)
    if viz_style==0: print('--- Variable is {}, classes are: {}'.format(xName, cat_tick_vals))
    
    df_outcome = defaultdict()
    tick_count_outcome = defaultdict()
    tick_percent_outcome = defaultdict()
    for iTick in range(nX):
        df_outcome[iTick]  = df_cat[yName] == outcome_tickNames[iTick]
        tick_count_outcome[iTick] = np.empty(n_ticks)
        tick_percent_outcome[iTick] = np.empty(n_ticks)


    #### Get Count of Each Category
    for tick_id, tick_name in enumerate(cat_tick_vals):
        try:
            this_class = df_cat[xName] == cat_tick_vals[tick_id]
            
            if this_class.sum() == 0:
                print('nothing in {} class and {} variable...'.format(tick_name, xName))
                continue

            for iTick in range(nX):
                this_class_count = (this_class & df_outcome[iTick]).sum()
                tick_percent_outcome[iTick][tick_id] = 100*this_class_count/this_class.sum()
                tick_count_outcome[iTick][tick_id] = this_class_count
            
        except:
            raise ValueError('Could not read the tick labels from the dictionary. Check the "ticksDictionary"...')
        
    #### Add Plot Bars
    try:
        for iTick in range(nX):
            plt.bar(cat_tick_vals, tick_percent_outcome[iTick], edgecolor = "black",
                    bottom = np.sum([tick_percent_outcome.get(key) for key in range(iTick)], axis=0),
                label='y{}'.format(iTick+1), color=col_list[iTick])
    except:
        raise ValueError('Could not visualize the bars, probably the number of "Ticks" is too many (more than 5)...')
    
    #### Add Labels to the Plots 
    try:
        plt.xlabel(rename_dict[xName], fontsize=label_fontsize)
        try:
            plt.ylabel(rename_dict[yName], fontsize=label_fontsize)
        except:
            pass
        plt.yticks([])
        
    except:
        plt.xlabel(xName[rename_index:].upper(), fontsize=label_fontsize)
        # plt.ylabel(yName[rename_index:].upper())
        plt.yticks([])
        
    
    #### Add Text to the Bars
    txt_font_size_list = [22, 22, 20, 18, 16, 12, 12, 10]
    txt_font_size = txt_font_size_list[n_ticks-1]

    txt_y = defaultdict()
    txt_y[0] = tick_percent_outcome[0]/2
    txt_y[1] = tick_percent_outcome[0] + tick_percent_outcome[1]/2
    if nX >=3:
        txt_y[2] = tick_percent_outcome[0] + tick_percent_outcome[1] + tick_percent_outcome[2]/2
    if nX >=4:
        txt_y[3] = tick_percent_outcome[0] + tick_percent_outcome[1] + tick_percent_outcome[2] +\
                    tick_percent_outcome[3]/2
    if nX >=5:
        txt_y[4] = tick_percent_outcome[0] + tick_percent_outcome[1] + tick_percent_outcome[2] +\
                    tick_percent_outcome[3] + tick_percent_outcome[4]/2
    if nX >=6:
        txt_y[5] = tick_percent_outcome[0] + tick_percent_outcome[1] + tick_percent_outcome[2] +\
                    tick_percent_outcome[3] + tick_percent_outcome[4] + tick_percent_outcome[5]/2                    
                    
    for iTick in range(nX):
        for xpos, ypos, percent, count in zip(cat_tick_vals, txt_y[iTick],
                                              tick_percent_outcome[iTick],
                                              tick_count_outcome[iTick]):
            
            if percent >= pcnt_control:
                if viz_style == 0:
                    plt.text(xpos, ypos, "{:.0f}%\n({:.0f})".format(percent, count),
                          ha="center", va="center", fontsize = txt_font_size, c= text_color, fontweight='bold')
                    
                elif viz_style == 1:
                    plt.text(xpos, ypos, "{:.0f}%".format(percent),
                          ha="center", va="center", fontsize = txt_font_size+2, c= text_color, fontweight='bold')        
                    
                elif viz_style == 2:
                    plt.text(xpos, ypos, "{:.0f}%\n(n = {:.0f})".format(percent, count),
                          ha="center", va="center", fontsize = txt_font_size, c= text_color, fontweight='bold')
                    
                elif viz_style == 3:
                    plt.text(xpos, ypos, "n = {:.0f}".format(count),
                          ha="center", va="center", fontsize = txt_font_size, c= text_color, fontweight='bold')
                    
                    
    return tick_percent_outcome



def calc_BreakDownGroups(data_wCatNames,  xName, yName, ticksDictionary, rename_index = 0, rename_dict = None, pcnt_control = 20):
    #### Data Prep
    df_cat = data_wCatNames.loc[:,[xName, yName]]
    
    outcome_tickNames = ticksDictionary[yName].split(', ')
    nX = len(outcome_tickNames)
    # print('\nOutcome is {}, classes are: {}'.format(yName, outcome_tickNames))
    
    cat_tick_vals = ticksDictionary[xName].split(', ')
    n_ticks = len(cat_tick_vals)
    # print('--- Variable is {}, classes are: {}'.format(xName, cat_tick_vals))
    
    df_outcome = defaultdict()
    tick_count_outcome = defaultdict()
    tick_percent_outcome = defaultdict()
    for iTick in range(nX):
        df_outcome[iTick]  = df_cat[yName] == outcome_tickNames[iTick]
        tick_count_outcome[iTick] = np.empty(n_ticks)
        tick_percent_outcome[iTick] = np.empty(n_ticks)


    #### Get Count of Each Category
    for tick_id, tick_name in enumerate(cat_tick_vals):
        try:
            this_class = df_cat[xName] == cat_tick_vals[tick_id]
            
            if this_class.sum() == 0:
                print('nothing in {} class and variable...'.format(tick_name))
                continue

            for iTick in range(nX):
                this_class_count = (this_class & df_outcome[iTick]).sum()
                tick_percent_outcome[iTick][tick_id] = 100*this_class_count/this_class.sum()
                tick_count_outcome[iTick][tick_id] = this_class_count
            
        except:
            raise ValueError('Could not read the tick labels from the dictionary. Check the "ticksDictionary"...')
            
    return tick_percent_outcome



#%% Age Matching
from scipy import stats # statistics toolbox
from scipy.stats import norm
import matplotlib.pyplot as plt

def age_match_dataframe(dataframe, target_name, col_name = 'Age', col_bins = ['<40', '40-70', '>70'], down_or_up = 'u', n_gen_ds = 10, replace_df = [], replace_df_oob = []):
    
    if col_name == 'Gender':
        col_name_bins = col_name
    else:
        col_name_bins = col_name+'_Bins'
        

    if replace_df:
        df_out = replace_df
    else:
        df_out = []
    col_name_bins = col_name+'_Bins'
    df_check = dataframe.copy(deep=True)
    
    df_chck_1 = df_check[df_check[col_name_bins] == col_bins[0]]#'0-18']
    df_chck_2 = df_check[df_check[col_name_bins] == col_bins[1]] #'18-50']
    df_chck_3 = df_check[df_check[col_name_bins] == col_bins[2]]#'50-65']
    # df_chck_4 = df_check[df_check['Age_Bins'] == '>65']
    
    
    df_chck_1_n = df_chck_1[df_chck_1[target_name] == 'No VTE']
    df_chck_1_y = df_chck_1[df_chck_1[target_name] == 'VTE']
    
    df_chck_2_n = df_chck_2[df_chck_2[target_name] == 'No VTE']
    df_chck_2_y = df_chck_2[df_chck_2[target_name] == 'VTE']
    
    df_chck_3_n = df_chck_3[df_chck_3[target_name] == 'No VTE']
    df_chck_3_y = df_chck_3[df_chck_3[target_name] == 'VTE']
    
    # df_chck_4_n = df_chck_4[df_chck_4['VTE_post_fracture'] == 'No VTE']
    # df_chck_4_y = df_chck_4[df_chck_4['VTE_post_fracture'] == 'VTE']
    
    n_to_y_ratio = 4
    if replace_df:
        oob_data = replace_df_oob
    else:
        oob_data = []
        
    p_list = []
    
    for seed in range(n_gen_ds):
        if down_or_up == 'd':
            df_chck_1_n_s = df_chck_1_n.sample(n=len(df_chck_1_y) * n_to_y_ratio, replace=True, random_state=seed)
            df_chck_2_n_s = df_chck_2_n.sample(n=len(df_chck_2_y) * n_to_y_ratio, replace=True, random_state=seed)
            df_chck_3_n_s = df_chck_3_n.sample(n=len(df_chck_3_y) * n_to_y_ratio, replace=True, random_state=seed)
            ##df_chck_4_n_s = df_chck_4_n.sample(n=len(df_chck_4_y) * n_to_y_ratio, replace=True, random_state=seed)
                
            df_check_new = pd.concat([df_chck_1_n_s, df_chck_1_y,\
                                      df_chck_2_n_s, df_chck_2_y,\
                                      df_chck_3_n_s, df_chck_3_y])#,\
                                      ## df_chck_4_n_s, df_chck_4_y])
        elif down_or_up == 'u':
                                  
            df_chck_1_y_s = df_chck_1_y.sample(n=len(df_chck_1_n) // n_to_y_ratio, replace=True, random_state=seed)
            df_chck_2_y_s = df_chck_2_y.sample(n=len(df_chck_2_n) // n_to_y_ratio, replace=True, random_state=seed)
            df_chck_3_y_s = df_chck_3_y.sample(n=len(df_chck_3_n) // n_to_y_ratio, replace=True, random_state=seed)
            
            df_check_new = pd.concat([df_chck_1_n, df_chck_1_y_s,\
                                      df_chck_2_n, df_chck_2_y_s,\
                                      df_chck_3_n, df_chck_3_y_s])#,\
        
            
        df_check_new = df_check_new.sample(frac=1) # shuffle everything
        oob_samples = set(dataframe.index.values) - set(df_check_new.index.values)


        #### Make sure Age is okay
        df_check_new_n = df_check_new[df_check_new[target_name] == 'No VTE']
        df_check_new_y = df_check_new[df_check_new[target_name] == 'VTE']
        
        _, pval = stats.ttest_ind(df_check_new_n[col_name], df_check_new_y[col_name], equal_var = False)
        p_list.append(pval)
                
        print('{} subjects are not included. {} subjects are included more than once.'.format(len(oob_samples), sum(df_check_new.index.value_counts()>1)))
    
    
        df_check_new.reset_index(drop=True, inplace=True)
        df_out.append(df_check_new)
        
        oob_data.append(oob_samples)
    
    return df_out, p_list, oob_data




def gender_match_dataframe(dataframe, target_name, col_name = 'Gender', col_bins = ['female', 'male'], down_or_up = 'u', n_gen_ds = 10, replace_df = [], replace_df_oob = []):

    if col_name == 'Gender':
        col_name_bins = col_name
    else:
        col_name_bins = col_name+'_Bins'

    if replace_df:
        df_out = replace_df
    else:
        df_out = []
        
    df_check = dataframe.copy(deep=True)
    
    df_chck_1 = df_check[df_check[col_name_bins] == col_bins[0]]
    df_chck_2 = df_check[df_check[col_name_bins] == col_bins[1]] 
    
    
    df_chck_1_n = df_chck_1[df_chck_1[target_name] == 'No VTE']
    df_chck_1_y = df_chck_1[df_chck_1[target_name] == 'VTE']
    
    df_chck_2_n = df_chck_2[df_chck_2[target_name] == 'No VTE']
    df_chck_2_y = df_chck_2[df_chck_2[target_name] == 'VTE']
    
    
    n_to_y_ratio = 4

    if replace_df:
        oob_data = replace_df_oob
    else:
        oob_data = []
        
    p_list = []
    
    for seed in range(n_gen_ds):
        if down_or_up == 'd':
            df_chck_1_n_s = df_chck_1_n.sample(n=len(df_chck_1_y) * n_to_y_ratio, replace=True, random_state=seed)
            df_chck_2_n_s = df_chck_2_n.sample(n=len(df_chck_2_y) * n_to_y_ratio, replace=True, random_state=seed)
                
            df_check_new = pd.concat([df_chck_1_n_s, df_chck_1_y,\
                                      df_chck_2_n_s, df_chck_2_y])
                
        elif down_or_up == 'u':
                                  
            df_chck_1_y_s = df_chck_1_y.sample(n=len(df_chck_1_n) // n_to_y_ratio, replace=True, random_state=seed)
            df_chck_2_y_s = df_chck_2_y.sample(n=len(df_chck_2_n) // n_to_y_ratio, replace=True, random_state=seed)
            
            df_check_new = pd.concat([df_chck_1_n, df_chck_1_y_s,\
                                      df_chck_2_n, df_chck_2_y_s])
        
            
        df_check_new = df_check_new.sample(frac=1) # shuffle everything
        oob_samples = set(dataframe.index.values) - set(df_check_new.index.values)


        #### Make sure Age is okay
        df_check_new_n = df_check_new[df_check_new[target_name] == 'No VTE']
        df_check_new_y = df_check_new[df_check_new[target_name] == 'VTE']
        
        # _, pval = stats.ttest_ind(df_check_new_n[col_name], df_check_new_y[col_name], equal_var = False)
        # p_list.append(pval)
                
        print('{} subjects are not included. {} subjects are included more than once.'.format(len(oob_samples), sum(df_check_new.index.value_counts()>1)))
    
    
        df_check_new.reset_index(drop=True, inplace=True)
        df_out.append(df_check_new)
        
        oob_data.append(oob_samples)
    
    return df_out, p_list, oob_data



#%%%%%    

################################
import seaborn as sns
import numpy as np

def show_dist_and_bins(dataframe, col_name = 'Age', col_labels = '<40, 40-70, >70',\
                       target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE',\
                       col_list=[['r'],['g'],['b'],['c'],['m'],['k']]):

    if col_name == 'Gender':
        col_name_bins = col_name
    else:
        col_name_bins = col_name+'_Bins'
        
    #### Visualize Binning Effect
    cat_dic = {target_name: target_labels, col_name_bins: col_labels}
    rename_dict = {target_name: 'Outcome', col_name: col_name}
    
    REVERSE = False

    plt.figure(dpi=500, figsize=(5,16))
    sns.set(style="whitegrid",
            rc={'axes.labelsize':22,
                'xtick.labelsize':14,
                'ytick.labelsize':16})

    if REVERSE:
        barplot_in_percent_general(dataframe,  target_name, col_name_bins, cat_dic, rename_dict = rename_dict, pcnt_control=5,col_list=col_list)

        plt.legend([leg for leg in cat_dic[col_name_bins].split(', ')], loc = 2, \
                                bbox_to_anchor= (1.0, .73),
                                shadow = True, fontsize = 16)
            
    else:
        barplot_in_percent_general(dataframe,  col_name_bins, target_name, cat_dic, rename_dict = rename_dict, pcnt_control=5,col_list=col_list)

        plt.legend(cat_dic[target_name].split(', '), loc = 2, \
                        bbox_to_anchor= (1.0, .73),
                        shadow = True, fontsize = 24)
    plt.ylim([0,100])
    plt.show()

        
    legend_labels = cat_dic[target_name].split(', ')
    if col_name == 'Age': # this has to be continuous
        ##### Visualize Data Distributioon
        Sorted_Date = dataframe[col_name].sort_values(ascending=True)#%%
        # print(Sorted_Date[:10])
        # print(Sorted_Date[~Sorted_Date.isnull()][-3:])
        age_diff = int(Sorted_Date[~Sorted_Date.isnull()].iloc[-1] - Sorted_Date.iloc[0])//6
    
        vte_True = np.where(dataframe[target_name] == target_yes, True, False)
    
        # Generate some data for this 
        # demonstration.
    
        data_n = dataframe[~vte_True][col_name].dropna().reset_index(drop=True)
        data_y = dataframe[vte_True][col_name].dropna().reset_index(drop=True)
    
        # data_y = data_y.sample(n=500, replace=True, random_state=1, weights='Age_Bins')
    
        # data_n = data_n[~((data_n>70) & (data_n<95))]
        # data_n = pd.concat((data_n,pd.Series([25]*40+[30]*40+[35]*80+[45]*80+[50]*20+[55]*5))).reset_index(drop=True)
    
        # Fit a normal distribution to
        # the data:
        # mean and standard deviation
        mu_n, std_n = norm.fit(data_n) 
        mu_y, std_y = norm.fit(data_y) 
        fig, ax = plt.subplots(dpi=300,figsize=(9,6))
        plt.hist(data_n, bins=age_diff, color='orange', density =True, label=legend_labels[0])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p_n = norm.pdf(x, mu_n, std_n)
        plt.plot(x, p_n, 'k', linewidth=2, label='_Hidden')
        plt.hist(data_y, bins=age_diff, color='blue', alpha=0.5, density =True, label=legend_labels[1])
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p_y = norm.pdf(x, mu_y, std_y)
        plt.plot(x, p_y, 'k', linewidth=2, label='_Hidden')
        plt.xlabel(col_name, fontsize = 20)
        plt.ylabel('Number of Patients', fontsize = 20)
        plt.legend(frameon=False, shadow = True, fontsize = 18)
        title = "{} {}: {:.2f} and {:.2f}\n{} {}: {:.2f} and {:.2f}"\
                                .format(legend_labels[0], target_name, mu_n, std_n, legend_labels[1], target_name, mu_y, std_y)
        plt.title(title)    
        
        
        
        
##########
def show_bins_outcome(dataframe, col_name = 'Age', col_labels = '<40, 40-70, >70',\
                       target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE',\
                           visualize = True):

    
    #### Visualize Binning Effect
    cat_dic = {target_name: target_labels, col_name: col_labels}
    rename_dict = {target_name: 'Outcome', col_name: col_name}
    
    REVERSE = False

    if visualize:
        plt.figure(dpi=500, figsize=(5,16))
        sns.set(style="whitegrid",
                rc={'axes.labelsize':22,
                    'xtick.labelsize':14,
                    'ytick.labelsize':16})

    if REVERSE:
        if visualize:
            tick_percent_outcome = barplot_in_percent_general(dataframe,  target_name, col_name, cat_dic, rename_dict = rename_dict, pcnt_control=5)
    
            plt.legend([leg for leg in cat_dic[col_name].split(', ')], loc = 2, \
                                    bbox_to_anchor= (1.0, .73),
                                    shadow = True, fontsize = 16)
        else:
            tick_percent_outcome = calc_BreakDownGroups(dataframe,  target_name, col_name, cat_dic, rename_dict = rename_dict, pcnt_control=5)
            
    else:
        if visualize:
            tick_percent_outcome = barplot_in_percent_general(dataframe,  col_name, target_name, cat_dic, rename_dict = rename_dict, pcnt_control=5)
    
            plt.legend(cat_dic[target_name].split(', '), loc = 2, \
                            bbox_to_anchor= (1.0, .73),
                            shadow = True, fontsize = 24)
                
        else:
            tick_percent_outcome = calc_BreakDownGroups(dataframe,  col_name, target_name, cat_dic, rename_dict = rename_dict, pcnt_control=5)
            
    if visualize:
        plt.ylim([0,100])
        plt.show()        

        
    return tick_percent_outcome
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        