# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:21:56 2021

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
from scipy import stats # statistics toolbox
import os
from itertools import chain, compress, combinations # to filter lists in one line
import matplotlib.pyplot as plt

import util_project as uprj
import util_stats as ustats

from tqdm import tqdm

from datetime import date

seed_val = 54

#%% Loading the datasets
import VTE_PathReader as prj

with open(prj.path_data_imputed, 'rb') as f:
    Data_imputed = pickle.load(f)
    Data_imputed.drop('MRN/EMPI', axis=1, inplace=True)

with open(prj.path_data_fullcase, 'rb') as f:
    Data_noNan = pickle.load(f)

with open(prj.path_data_humanread, 'rb') as f:
    data_wName = pickle.load(f)
    
    
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
    

# cat_dic, cat_len = uprj.get_cat_data(prj.path_description, all_cat_vars, 'ankle_fracture')

# Data_imputed.drop('US_positive_for_post_fracture_VTE', axis=1, inplace=True)
# Data_imputed.drop('Post_fracture_PE', axis=1, inplace=True)

# Data_imputed = Data_imputed[Data_imputed.Previous_DVT==0]
# Data_imputed = Data_imputed[Data_imputed.Previous_PE==0]
# Data_imputed.reset_index(drop = True, inplace = True)
# Data_imputed.drop('Previous_DVT', axis=1, inplace=True)
# Data_imputed.drop('Previous_PE', axis=1, inplace=True)

    

cat_dic, cat_len = uprj.get_cat_data(prj.path_description, all_cat_vars, 'ankle_fracture')

# for var_name in ['Post_fracture_PE']:#,'Previous_DVT','Previous_PE']:
#     del cat_dic[var_name]
#     del cat_len[var_name]

tick_list = list(cat_dic.items())

outcomes_varnames = cat_dic[outcome_cat_vars].split(', ')


key_vte_lab = [key for key in Data_imputed.columns if '_vte_' in key]
for key_vte in key_vte_lab:
    Data_imputed.drop(key_vte,  axis=1, inplace=True)
    

#%%
# # print(data['Healing_Status'].unique())


# # ### let's remove delayed
# # data.replace({'Healing_Status':{1: np.nan}}, inplace = True) # remove delayed
# # data.replace({'Healing_Status':{2: 1}}, inplace = True) #change union to 1

# # data.dropna(subset=['Healing_Status'], how='any', inplace=True)

# # print(data['Healing_Status'].unique())

# # cat_dic['Healing_Status'] = 'nonunion, union'
# #### REMOVE HEIGHT AND WEIGHT

# del(predictor_vars[1])
# del(predictor_vars[1])
# data.drop('Weight', axis=1, inplace=True)
# data.drop('Height', axis=1, inplace=True)

#%%
#% Variables Definition

Medications = [key for key, _ in Data_imputed.items() if 'med_' in key]
Comorbidity = [key for key, _ in Data_imputed.items() if 'cm_' in key]
# Past_Medical = [key for key, _ in Data_imputed.items() if 'past_mh_' in key] 
Prophylaxis = [key for key, _ in Data_imputed.items() if 'prphlyx_' in key] 


# Lab_Data_At_Fx = [key for key, _ in Data_imputed.items() if '_fra_lab' in key] #chronic disease
Lab_Data_At_Fx = [key for key, _ in Data_imputed.items() if 'lab_' in key] #chronic disease


Target_Name = outcome_vars[0] #['Healing_Time','Healing_Status']


#%% Date Manipulation
bins = [0, 40, 70, 150]
 # (<18, 18-50, 50-65, >65)
labels = [0, 1, 2] #['0-18yrs', '18-44 yrs', '44-54 yrs', '55-64 yrs','65-74 yrs','75> yrs']
Data_imputed['Age_Bins'] = pd.cut(Data_imputed.Age, bins= bins, labels=labels, include_lowest = True).astype(int)
cat_dic['Age_Bins'] = '<40, 40-70, >70'
cat_len['Age_Bins'] = 3
# rename_dict['Age_Bins'] = 'Age'



bins = [0, 18.5, 25, 30, 100]
labels = [0,1,2,3]#['underweight', 'healthy', 'overweight', 'obese']
Data_imputed['BMI_Bins'] = pd.cut(Data_imputed.BMI, bins= bins, labels=labels, include_lowest = True).astype(int)
cat_dic['BMI_Bins'] = 'Under-W, Healthy, Over-W, Obese'
cat_len['BMI_Bins'] = 4


tick_list = list(cat_dic.items())


Comorbidity = [key for key in Comorbidity if key not in ['cm_dvt']]
# Past_Medical = [key for key in Past_Medical if key not in ['past_mh_dvt','past_mh_pulmonary embolism']]



bins = [0, 2, 100]
labels = [0,1]#['underweight', 'healthy', 'overweight', 'obese']
Data_imputed['CCI_Bins'] = pd.cut(Data_imputed.CCI, bins= bins, labels=labels, include_lowest = True).astype(int)
cat_dic['CCI_Bins'] = '≤2, >2'
cat_len['CCI_Bins'] = 2



####%% Change the variable of the outcome name to its string format
data_stats = Data_imputed.copy(deep=True)

data_stats[outcome_cat_vars] = data_stats[outcome_cat_vars].replace(to_replace=np.sort(data_stats[outcome_cat_vars].unique()),
                                                        value=cat_dic[outcome_cat_vars].split(', '))




#%% Make Categorical data nice and neat!


# #%% Statistical Analysis - Comparing columns (1-by-1): this is for practice, you should account for multiple variance always in large datasets
# t, p_val = stats.ttest_1samp(data['BMI'], 0) # 1-sample t-test
# print('p-val is: {:.2e}'.format(p_val))

# female_ht = eda_data[eda_data['Gender'] == 'female']['Healing_Time']
# male_ht = eda_data[eda_data['Gender'] == 'male']['Healing_Time']
# t, p_val = stats.ttest_ind(female_ht, male_ht) # 2-sample t-test (independent)
# print('p-val is: {:.2e}'.format(p_val))

# t, p_val = stats.ttest_rel(eda_data['HBA1C_Level'],eda_data['WBC_Level']) # 2-sample paired t-test
# print('p-val is: {:.2e}'.format(p_val))

# #T-tests assume Gaussian errors. We can use a Wilcoxon signed-rank test, that relaxes this assumption:

# t, p_val = stats.wilcoxon(eda_data['HBA1C_Level'],eda_data['WBC_Level']) # 2-sample paired t-test
# print('p-val is: {:.2e}'.format(p_val))


# #%% Statistical Analysis - Linear Models: ordinary least squares (OLS)
# from statsmodels.formula.api import ols

# model = ols("Healing_Status ~ C(Fracture_Zone)+C(Osteoprosis)+1", data).fit()
# print(model.summary())


# groupby_gender = eda_data.groupby('Gender')
# for gender, value in groupby_gender['Healing_Time']:
#     print((gender, value.mean()))
# #%%

# import statsmodels.api as sa
# import statsmodels.formula.api as sfa
# import scikit_posthocs as sp


# sp.posthoc_ttest(df, val_col='SepalWidth', group_col='Species', p_adjust='holm')
match_data = True
dataset_name = 'All'
if dataset_name == 'All':
    print('*** Loading matched data for all variables [EDA and Staistical Analysis]')
    path_matched = prj.path_data_age_gender_matched_all
else:
    print("--> Looking at Training data only....")
    path_matched = prj.path_data_age_gender_matched_train

if match_data:
    with open(path_matched, 'rb') as f:
        df_match_all = pickle.load(f)
        df_big = pickle.load(f)
            
        
#%% Age Matching - Run this or load data
run_this = True
today = date.today()
day_str = today.strftime("%m%d%Y")

if match_data and run_this:
    significant_ORs = pd.DataFrame()
    for counter, df_match in enumerate(df_match_all):
        # df_big = pd.concat((df_big, df_match)) 
        
        df_age_matched = df_match.copy(deep=True)
        
        # ustats.show_dist_and_bins(temp_df, col_name = 'Age', col_labels = '<40, 40-70, >70',\
        #                         target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE')
        
        # ustats.show_dist_and_bins(temp_df, col_name = 'Age', col_labels = '0, 1, 2',\
        #                         target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE')


        # Convert "Outcome Strings (e.g., No VTE/VTE)" to Categorical Vars (0, 1)
        df_age_matched[outcome_cat_vars] = df_age_matched[outcome_cat_vars].replace(to_replace=np.sort(df_age_matched[outcome_cat_vars].unique()),
                                                                value=[0,1])


        df_age_matched.drop(columns=['Age', 'BMI', 'CCI', 'Hospital_stay', 'trauma_to_treatment_days'], inplace=True)
        
        
        _, ORs, _ = ustats.run_ChiSquared(df_age_matched, 'VTE_post_fracture', print_on = False, alpha = 0.05)
        
        
        
        # for key in ORs.index:
            # print(ORs.loc[key][['OR', 'p-value','CI Value']])
            # significant_ORs[key]
            
        significant_ORs = pd.concat([significant_ORs, ORs])
        # print('\n* p-value of 0 means <0.0001')

        if counter % 500 == 0:
            print(f'{counter} iteration elapsed...')
        
        counter += 1

    # else:
    #     temp_df = data_wName.copy(deep=True)
            
            
    #     ustats.show_dist_and_bins(temp_df, col_name = 'Age', col_labels = '<40, 40-70, >70',\
    #                             target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE')
            
            
    significant_ORs.reset_index(inplace=True)    
    
    ####
    # oob_complete_set = set()
    # oob_counter = {}
    # oob_counter[list(oob_samples[i])]
    
    # rm_data_len = []
    # for i in range(len(oob_samples)):
    #     # print(len(oob_samples[i]))
    #     rm_data_len.append(len(oob_samples[i]))
        
    #     oob_temp = {index: oob_counter.get(index, 0)+1 for index in list(oob_samples[i])}
        
    #     oob_counter.update(oob_temp)
        
    #     oob_complete_set = oob_complete_set.union(oob_samples[i])
    
    # print(len(oob_complete_set))
    # print(sum(data_stats['VTE_post_fracture']=='VTE'))
    
    # print('{:.0f} (Range: {} to {}) subjects were removed on average during sampling.'.format(np.mean(rm_data_len), np.min(rm_data_len), np.max(rm_data_len)))
    
    # min(oob_counter.values())
    # max(oob_counter.values())
    # np.median(list(oob_counter.values()))

    path_age_matched = os.path.join(prj.path_data_folder,f'VTE_AgeGenderMatched_{day_str}.npz')
    np.savez(path_age_matched, name2=significant_ORs)
    
    
else:
    print('loading already saved data....')
    raise ValueError('Run This Again!')
    df_all = np.load('VTE_AgeGenderMatched_10052022.npz', allow_pickle=True) # 'vte_resampled_data_age_gender.npz', allow_pickle=True)
    
    significant_ORs = pd.DataFrame(df_all['name2'], columns=['index','OR', '95% CI', 'p-value', 'CI Value'])
    # df_match_all = df_all['name4']



#%% Odds Ratio Calculation
ORs_group = significant_ORs.groupby('index')

full_sigs = []
for key, data in ORs_group:
    data = data.replace([np.inf, -np.inf, 'inf'], np.nan)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    if key != 'VTE_post_fracture' and len(data)>0:
        
        odds_ratio = round(np.nanmean(data['OR'].values.astype(float)),3)

        mean = round(np.nanmean(data['p-value'].values.astype(float)),3)
        sd   = round(np.nanstd(data['p-value'].values.astype(float)),3)
        count = len(data['p-value'])
        
        odds_ratio_ci = np.nanmean(data['CI Value'].values.astype(float))

        or_CI = '{:.2f} - {:.2f}'.format(odds_ratio-odds_ratio_ci, odds_ratio+odds_ratio_ci)


        full_sigs.append([key, odds_ratio, or_CI, mean, sd, count])


df_full_sigs = pd.DataFrame(full_sigs, columns=['Variable', 'Average OR', 'Average 95% CI', 'Average p-value', 'SD p-value', 'CountedAsSignificant'])
df_full_sigs = df_full_sigs.sort_values(by='CountedAsSignificant',ascending=False)


df_full_sigs['CI_Check'] = df_full_sigs['Average 95% CI'].apply(lambda x: not (float(x.split(' -')[1])>1 and float(x.split(' -')[0])<1))

df_full_sigs[['Average 95% CI','CI_Check']]

print(df_full_sigs[(df_full_sigs.CountedAsSignificant>5000) & (df_full_sigs['CI_Check'])].to_markdown())


results_table = df_full_sigs[(df_full_sigs.CountedAsSignificant>5000) & (df_full_sigs['CI_Check'])]


#%% Calculate Included Variables from Age-Matched Data
included = results_table['Variable'].values
print(included)
# df_all = np.load('vte_resampled_data.npz', allow_pickle=True)
# df_full_sigs = pd.DataFrame(df_all['name3'], columns=['Variable', 'OR', 'P-Val Mean', 'P-Val SD', 'Sig Count'])

# significant_ORs = pd.DataFrame(df_all['name2'], columns=['index','OR', '95% CI', 'p-value', 'CI Value'])



#%% 
predictor_vars_new = [key for key in data_stats.columns if key != 'VTE_post_fracture'] 


check_col   = 'MVA'
check_label = ['no', 'yes']

values = []
counter = 0
significant_GLMs = pd.DataFrame()
for df_check_new in tqdm(df_match_all):
    df_age_matched = pd.DataFrame(df_check_new, columns=data_stats.columns)

    
    # df_age_matched['lab_rbc'] = df_age_matched['lab_rbc'].replace(to_replace=[0, 1, 2],
    #                                                         value=['low', 'normal', 'high'])
    
    
    df_age_matched[check_col] = df_age_matched[check_col].replace(to_replace=list(range(len(check_label))),
                                                            value=check_label)
    
    tick_percent_outcome = ustats.show_bins_outcome(df_age_matched, col_name = check_col, col_labels = ', '.join(val for val in check_label),\
                            target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE', visualize=False)

        
    values.append(list(tick_percent_outcome[0]))
    
    counter += 1
    # if counter % 500 == 0:
    #     print('Iteration {}'.format(counter))
        
    # df_age_matched.drop(columns=['Age_Bins', 'BMI_Bins', 'CCI_Bins'], inplace=True)

    # results_df, glm_fit = ustats.run_GLM(df_age_matched, Target_Name, predictor_vars_new, alpha = 0.05)

    # significant_GLMs = pd.concat([significant_ORs, ORs])

check_df = pd.DataFrame(values, columns=check_label)


print(check_df.describe())

#%% Continuous Variables - Matched Cohort
# variable definitions
continuous_variables = predictor_cont_vars + ['CCI']

# for-loop to calculate the ANOVA and the table for each matched cohort
anova_results = []
for df_check_new in tqdm(df_match_all):
    df_age_matched = pd.DataFrame(df_check_new, columns=data_stats.columns)


    _, access_tbl = ustats.run_OneWayANOVA(df_age_matched, continuous_variables, Target_Name, outcomes_varnames, print_on=False)

    anova_results.append(access_tbl)

# concatenate all of them in one dataframe to do summary statistics easier
df_anova_results = pd.concat(anova_results)

# for-loop over each variable and create a final table for the manuscript
anova_table = {} # this will have the final table as a list
for temp_var in continuous_variables:
    temp_table  = []
    
    temp_mean_no = df_anova_results.loc[temp_var]['No VTE_Mean'].mean()
    temp_sd_no   = df_anova_results.loc[temp_var]['No VTE_SD'].mean()
    temp_table.append('{:.1f} ± {:.1f}'.format(temp_mean_no, temp_sd_no))

    temp_mean_yes = df_anova_results.loc[temp_var]['VTE_Mean'].mean()
    temp_sd_yes   = df_anova_results.loc[temp_var]['VTE_SD'].mean()
    temp_table.append('{:.1f} ± {:.1f}'.format(temp_mean_yes, temp_sd_yes))
    
    # get the average p-value
    temp_table.append('{:.3f} ({:.3f})'.format(df_anova_results.loc[temp_var]['p-val'].mean(),
                                               df_anova_results.loc[temp_var]['p-val'].std()))

    # get the number of times we have had significant difference
    temp_table.append(sum(df_anova_results.loc[temp_var]['p-val'] < 0.05))

    # make the list into a dictionary
    anova_table[temp_var] = temp_table

#convert the dictionary to a dataframe
anova_dataframe = pd.DataFrame(anova_table, index = outcomes_varnames+['Average (SD) p-val', 'Counted As Significant']).T




#%%

# df_big = pd.DataFrame()

# for df_match in tqdm(df_match_all):
#     temp_df = pd.DataFrame(df_match, columns=data_stats.columns)

#     df_big = pd.concat((df_big, temp_df)) # concatenate everything to a big dataframe for visualization of the fully matched dataset

# sig_parameters = df_full_sigs.Variable[df_full_sigs.CountedAsSignificant>2000].values

#%%
check_col   = 'Smoking'#'lab_k'#'MVA'
check_label = ['no', 'yes', 'former']

#['conservative', 'operative']
#['low','normal','high']
#['no', 'yes']

var_no_vte = []
var_yes_vte = []
counter = 0
for df_match in tqdm(df_match_all):
    # temp_df = pd.DataFrame(df_match, columns=data_stats.columns)
    temp_df = df_match.copy(deep=True)
    temp_df[check_col] = temp_df[check_col].replace(to_replace=list(range(len(check_label))),
                                                            value=check_label)
    
    temp_val = ustats.show_bins_outcome(temp_df, col_name = check_col, col_labels = ', '.join(val for val in check_label),\
                            target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE', visualize=False)
        
    var_no_vte.append(temp_val[0])
    var_yes_vte.append(temp_val[1])  
    
    counter += 1
    
var_df = pd.DataFrame(var_no_vte, columns=check_label)

print(var_df.mean())
# var_df.std() 


import seaborn as sns


plt.figure(figsize=(6, 6), dpi=600)
ax = sns.barplot(data=var_df, estimator=np.mean, ci='sd', capsize=.1)
ax.set_ylim([0,100])


#%% TEMP: another approach to do this
df_big[check_col] = df_big[check_col].replace(to_replace=list(range(len(check_label))),
                                                        value=check_label)

temp_val = ustats.show_bins_outcome(df_big, col_name = check_col, col_labels = ', '.join(val for val in check_label),\
                        target_name = 'VTE_post_fracture', target_labels = 'No VTE, VTE', target_yes = 'VTE', visualize=False)
    
var_no_vte_big  = temp_val[0]
var_yes_vte_big = temp_val[1]

var_df_big = pd.DataFrame(var_no_vte_big, index=check_label).T

plt.figure(figsize=(6, 6), dpi=600)
ax = sns.barplot(data=var_df_big)
ax.set_ylim([0,100])



#%% Not Age-Matched ANOVA
from scipy.stats import f_oneway

from statsmodels.stats.multicomp import pairwise_tukeyhsd


statistics_table =  ustats.run_OneWayANOVA(data_stats, continuous_variables, Target_Name, outcomes_varnames)






#%%
# eda_data = temp_df.copy(deep=True)


lab_name_dictionary = {key: key.split('_')[0].upper() for key in Lab_Data_At_Fx}

from collections import defaultdict

varNames =  [key for key in Data_imputed.keys() if key != 'VTE_post_fracture'\
                                                     and key != 'Post_fracture_PE'\
                                                         and ('PT_INR' not in key)\
                                                             and ('PT_AT' not in key)\
                                                                 and ('Hospital_stay' not in key)\
                                                                     and ('Age' not in key)\
                                                                         and ('BMI' not in key)\
                                                                             and ('CCI' not in key)\
                                                                                 and ('treatment_to_trauma' not in key)]
varNames = varNames + ['Age_Bins','BMI_Bins','CCI_Bins']

# ['Gender','Treatment_of_fracture', 'Age_Bins','Race',
#             'VTE_Prophylaxis','Previous_PE','Previous_DVT',
#             'Multiple_fracture','Wound_type']#['Thyroid_Disease','Gender','Race','Activity_Level','Smoking','Race_New']

full_table_cat = defaultdict()
table_chi2 = defaultdict()
table_pearson = defaultdict()

for varName in varNames:    
    chi2_crosstab = pd.crosstab(data_wName[varName], data_wName[Target_Name], margins=True)    
    chi2, chi2_pval, _, _ = stats.chi2_contingency(chi2_crosstab)

    pearson_r, pearson_pval = stats.pearsonr(Data_imputed[varName], Data_imputed[Target_Name])

    try:
        label_names = cat_dic[varName].split(', ')
    except:
        if 'lab' == varName[-3:]:
            lab_name = varName.split('_')[0]
            label_names = prj.lab_tick_names[lab_name]

            
        elif 'lab' == varName[:3]:
            label_names = ['low', 'normal', 'high']
            
        else:
            label_names = ['Yes', 'No']
        
    for varType in label_names:
        count_set = []
        for case in outcomes_varnames:                    
            this_set = data_wName.loc[data_wName[Target_Name] == case, varName]
            
            count_set.append(np.sum(this_set==varType))
            
                                             
        ratio_per_outccome = np.array(count_set/sum(count_set))*100
        n_for_cat = sum(count_set)
        # print(n_for_cat)
        full_table_cat[varName+'_'+varType+' (n = {})'.format(n_for_cat)] = ['{:.0f}%'.format(ratio_per_outccome[0]), '{:.0f}%'.format(ratio_per_outccome[1])]
    
    table_chi2[varName] = chi2_pval
    table_pearson[varName] = [pearson_r, pearson_pval]


cat_table = pd.DataFrame(full_table_cat, index = outcomes_varnames).T
print(cat_table)


chi2_table = pd.Series(table_chi2)
pearson_table = pd.DataFrame(table_pearson, index=['r','p-val']).T
print(pearson_table)





#%% Let's do some chi-square
# new_data = Data_imputed.copy(deep=True)
new_data = data_stats.copy(deep=True)

new_data[outcome_cat_vars] = new_data[outcome_cat_vars].replace(to_replace=np.sort(data_stats[outcome_cat_vars].unique()),
                                                        value=[0,1])
# new_data = new_data[varNames+[Target_Name]]
new_data.drop(columns=['Age', 'BMI', 'CCI', 'Hospital_stay', 'trauma_to_treatment_days'], inplace=True)


_, ORs, _ = ustats.run_ChiSquared(new_data, 'VTE_post_fracture', print_on = True, alpha = 0.05)
# sig_corr_chi, ORs, CrossTabs = ustats.run_ChiSquared(new_data, Target_Name, print_on = True, alpha = 0.1)

print('\n* p-value of 0 means <0.0001')






#%% Pearson's R
sig_corr_pearson = ustats.run_PerasonTest(new_data, Target_Name, print_on = True, alpha = 0.05)





#%% Generalized Linear Models

# predictor_vars_new = [prd for prd in predictor_vars if prd not in ['Post_fracture_PE', 'Positive_CT_scan',
#                                                          'US_positive_for_post_fracture_VTE',
#                                                          'Previous_DVT', 'Previous_PE',
#                                                          'cd_dvt','past_mh_pulmonary embolism']]

# predictor_vars_new = [key for key in new_data.columns if 'statin' in key] 

prphlyx_keys = [key for key in new_data.columns if key != 'VTE_post_fracture' and 'prphlyx' in key] 



predictor_vars_new = prphlyx_keys + ['med_statin']


results_df, glm_fit = ustats.run_GLM(new_data, Target_Name, predictor_vars_new, alpha = 0.1)


# cd_hypercholesterolemia               0.3608      0.146      2.475      0.013       0.075       0.646
# VTE_Prophylaxis                      -0.3119      0.087     -3.569      0.000      -0.483      -0.141
# Post_fracture_PE                      0.4036      0.062      6.524      0.000       0.282       0.525


#%%


import seaborn as sns


g = sns.catplot(x="med_statin", y='prphlyx_warfarin',
                data=new_data, height=2.5, aspect=.8)



#%% Spearmans R
mod_vars = [pr for pr in predictor_vars]# if not pr in \
TestingParameters = [Target_Name] + mod_vars #mod_vars #Chronic_Diseases #+['Healing_Status']


new_data = Data_imputed.copy(deep=True)
new_data = new_data[TestingParameters]

print('\n\nUsing Spearmans R:')
r_vals, p_vals = stats.spearmanr(new_data)
    
crit = p_vals[0, 1:] < 0.05
sig_names = list(compress(TestingParameters[1:], crit))
sig_r = list(compress(r_vals[0,1:], crit))
sig_p = list(compress(p_vals[0,1:], crit))

sig_corr_spearman = pd.DataFrame(zip(sig_r,sig_p), index=sig_names, columns=['R-Adj','p-val'])

print(sig_corr_spearman)


print('\n\nUsing Point biserial correlation for Cntn and Kendal Tau for Catgs:')
tau = np.empty((len(TestingParameters), ))
p_value = np.empty((len(TestingParameters), ))
for cat_id, cat_name in enumerate(TestingParameters):
    if cat_name in predictor_cont_vars:
        tt, pp = stats.pointbiserialr(new_data.iloc[:,0], new_data.iloc[:,cat_id])
        tau[cat_id] = tt
        p_value[cat_id] = pp 
    else:
        tt, pp = stats.kendalltau(new_data.iloc[:,0], new_data.iloc[:,cat_id])
        tau[cat_id] = tt
        p_value[cat_id] = pp

crit = p_value < 0.05
sig_names = list(compress(TestingParameters, crit))
sig_r = tau[crit]
sig_p = p_value[crit]

sig_corr = pd.DataFrame(zip(sig_r,sig_p), index=sig_names, columns=['R-Adj','p-val'])
print(sig_corr[1:])





#%% Decision Trees
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics, tree #Import scikit-learn metrics module for accuracy calculation

Data_modeling = Data_imputed.copy(deep=True)
Data_modeling = Data_modeling[Data_modeling['VTE_Prophylaxis']==0]
# Data_modeling = Data_modeling[Data_modeling['Previous_PE']==0]
# Data_modeling = Data_modeling[Data_modeling['Previous_DVT']==0]

# Data_modeling.drop(columns=[key for key in Data_modeling if '_Bins' in key], inplace=True)

Feature_Names = [key for key in Data_modeling.keys() if key !=Target_Name\
                                                         and key!='Age' and key!='CCI'\
                                                         and key!='BMI' and key!='Hospital_stay'
                                                         and key!='Previous_DVT' and key!='Previous_PE']

    
    
# Feature_Names = [key for key in sig_corr_chi.index.tolist() if '_Bins' not in key]
# Feature_Names = [key for key in Feature_Names if key !=Target_Name and key!='Age'and key!='CCI']

# Feature_Names = ['Multiple_fracture','Treatment_of_fracture', 'Age',
#                  'Wound_type','Gender']
X = Data_modeling[Feature_Names] # Features
y = Data_modeling[Target_Name] # Target variable


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=44, max_depth = 5)

# Train Decision Tree Classifer
clf = clf.fit(X, y)

y_pred = clf.predict(X)

fig = plt.figure(figsize=(30,10), dpi= 300)
out = tree.plot_tree(clf, 
                   feature_names=Feature_Names,  
                   class_names = outcomes_varnames, fontsize= 16,
                   filled=True, label='none', impurity=False)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('k')
        arrow.set_linewidth(2)


###%% Accuracy Measures


y_pred_prob = clf.predict_proba(X)
y_pred_prob = y_pred_prob[:,1]

confusion_matrix = ustats.calculate_metrics(y, y_pred, y_pred_prob)


# FP = sum(y[y_pred==1] != 1)
# FN = sum(y[y_pred==0] != 0)



#%% Tree Feature Importance

f_importance = clf.feature_importances_

# sort them based on the value
FI_Val, FI_X = list(zip(*sorted(list(zip(f_importance, X.columns)), reverse=False)))


top_n_feature = 20

# create a figure
fig = plt.figure(figsize=(6, 10)) 
ax = fig.add_subplot(111)

# create a bar plot showing feature importance
plt.barh(FI_X[-top_n_feature:], FI_Val[-top_n_feature:])

# set the title
plt.title('Feature importance',
          loc='left', 
          fontsize=16)

# eliminate the frame from the plot
spine_names = ('top', 'right', 'bottom', 'left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)






#%% Correlation for Healing Time
Target_Name = outcome_vars[0]

healed_data = Data_imputed.copy(deep=True)
healed_data = healed_data[healed_data[Target_Name]==1]
healed_data.reset_index(drop=True, inplace = True)

data_size = healed_data.shape[1]

PR_r_vals = np.empty((data_size))
PR_p_vals = np.empty((data_size))
chi2 = np.empty((data_size))
p_value = np.empty((data_size))
dof = np.empty((data_size))

for iCol in range(0,data_size):
    ### Pearson R
    PR_r_vals[iCol], PR_p_vals[iCol] = stats.pearsonr(healed_data['Diabetes'], healed_data.iloc[:,iCol])
    
    

# stats.chi2_contingency(observed= observed)
# associations(eda_data)

print('\n\nUsing Persons R:')    
crit = PR_p_vals < 0.05
sig_PR_names = list(compress(healed_data.keys(), crit))
sig_PR_r = list(compress(PR_r_vals, crit))
sig_PR_p = list(compress(PR_p_vals, crit))

sig_corr_pearson = pd.DataFrame(zip(sig_PR_r,sig_PR_p), index=sig_PR_names, columns=['R-Adj','p-val'])

print(sig_corr_pearson)



#%%
# condition = healed_data['Osteoprosis'] != 1
condition = ['Treatment']

# plt.scatter(healed_data[condition][output_var], healed_data[condition]['Healing_Time'])


# plt.boxplot(healed_data[condition], healed_data['Healing_Time'])


boxplot = healed_data.boxplot(column=['HBA1C_Level'], by = condition, grid=False)



#%%% K-Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


kmeans = KMeans(
   init="random",
   n_clusters=2,
   n_init=20,
   max_iter=500,
   random_state=seed_val
   )



scaler = StandardScaler()
scaled_features = scaler.fit_transform(new_data.iloc[:,1:])
   
   
# A list holds the SSE values for each k
sse = []
kmeans.fit(scaled_features)
# print(kmeans.cluster_centers_)
print(kmeans.n_iter_)
# print(kmeans.labels_)



# plt.plot(range(1, 20), sse)
# plt.xticks(range(1, 20))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()

acc = accuracy_score(new_data.iloc[:,0] , kmeans.labels_)
cm = confusion_matrix(new_data.iloc[:,0] , kmeans.labels_)
print("Classification Accuracy: %{}".format(acc*100))
print("Confusion Matrix:")
print(cm)


#%% CART

import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report, confusion_matrix # for model evaluation metrics
from sklearn import tree # for decision tree models

import plotly
import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization
import graphviz # for plotting decision tree graphs


from collections import Counter
from imblearn.under_sampling import RandomUnderSampler



def fitting(X, y, criterion, splitter, mdepth, clweight, minleaf):

    
    ### Undersampling
    # print("Before undersampling: ", Counter(y))
    # undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=seed_val) #### majority or not minority
    # X, y = undersample.fit_resample(X, y)
    # print("After undersampling: ", Counter(y))

    
    ###
    
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_val)
    # X_train, y_train = X, y
    # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion, 
                                        splitter=splitter, 
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf, 
                                        random_state=seed_val, 
                                        # min_samples_split =5
                                  )
    clf = model.fit(X_train, y_train)

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    # Tree summary and model evaluation metrics
    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of leaves: ', clf.tree_.n_leaves)
    print('No. of features: ', clf.n_features_)
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print(confusion_matrix(y_test, pred_labels_te))
    print('--------------------------------------------------------')    
    print("")
    
    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')
    print('------------------Confusion Matrix----------------------')
    print(confusion_matrix(y_train, pred_labels_tr))
    
    
    # Use graphviz to plot the tree
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns, 
                                class_names=[str(list(clf.classes_)[0]), str(list(clf.classes_)[1])],
                                filled=True, 
                                rounded=True, 
                                #rotate=True,
                               ) 
    graph = graphviz.Source(dot_data)
    
    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf, graph


def Plot_3D(X, X_test, y_test, clf, x1, x2, mesh_size, margin):
            
    # Specify a size of the mesh to be used
    mesh_size=mesh_size
    margin=margin

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
            
    # Calculate predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    fig = px.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
                     opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with CART Prediction Surface",
                      paper_bgcolor = 'white',
                      scene = dict(xaxis=dict(title=x1,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(title=x2,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(title='Probability of Rain Tomorrow',
                                              backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0', 
                                              )))
    
    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='CART Prediction',
                              colorscale='Jet',
                              reversescale=True,
                              showscale=False, 
                              contours = {"z": {"show": True, "start": 0.5, "end": 0.9, "size": 0.5}}))
    fig.show()
    return fig

######3

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

def ensemble_fitting(X, y):

    # Instantiate lr
    lr = LogisticRegression(max_iter=500, random_state=seed_val)
    
    # Instantiate knn
    knn = KNN(n_neighbors=4)
    
    # Instantiate dt
    dt = tree.DecisionTreeClassifier(criterion='gini', 
                                        splitter='random', 
                                        max_depth=4,
                                        class_weight='balanced',
                                        min_samples_leaf=0.01,
                                        random_state=seed_val)
    
    # Define the list classifiers
    classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
    
    ### Undersampling
    print("Before undersampling: ", Counter(y))
    undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=seed_val) #### majority or not minority
    X, y = undersample.fit_resample(X, y)
    print("After undersampling: ", Counter(y))
    ###
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed_val)
    
    
    # Iterate over the pre-defined list of classifiers
    for clf_name, clf in classifiers:    
      
        # Fit clf to the training set
        clf.fit(X_train, y_train)    
      
        # Predict y_pred
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
      
        # Evaluate clf's accuracy on the test set
        print('{:s} : {:.3f}'.format(clf_name, accuracy))



    # Instantiate a VotingClassifier vc 
    vc = VotingClassifier(estimators=classifiers)     
    
    # Fit vc to the training set
    vc.fit(X_train, y_train)   
    
    # Evaluate the test set predictions
    pred_y_train = vc.predict(X_train)
    y_pred = vc.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print('Voting Classifier: {:.3f}'.format(accuracy))
    
    # clf = model.fit(X_train, y_train)

    # # Predict class labels on training data
    # pred_labels_tr = model.predict(X_train)
    # # Predict class labels on a test data
    # pred_labels_te = model.predict(X_test)
    
    print('*************** Evaluation on Training Data ***************')
    print('Accuracy Score: ', accuracy_score(y_train, pred_y_train))
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_y_train))
    print('--------------------------------------------------------')
    print('------------------Confusion Matrix----------------------')
    print(confusion_matrix(y_train, pred_y_train))
    
    
    print('*************** Evaluation on Test Data ***************')
    print('Accuracy Score: ', accuracy)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------')
    print(confusion_matrix(y_test, y_pred))
    print('--------------------------------------------------------')    
    print("")
    
    
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def rf_fitting(X, y):

    ### Undersampling
    # print("Before undersampling: ", Counter(y))
    # undersample = RandomUnderSampler(sampling_strategy='majority', random_state=seed_val) #### majority or not minority
    # X, y = undersample.fit_resample(X, y)
    # print("After undersampling: ", Counter(y))
    ###
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_val)

            
    # Fit rf to the training set    
    # rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=5,
    #         max_features='auto', max_leaf_nodes=None,
    #         min_impurity_decrease=0.0, min_impurity_split=None,
    #         min_samples_leaf=0.01, min_samples_split=2,
    #         min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
    #         oob_score=False, random_state=seed_val, verbose=0, warm_start=False)
    dt = tree.DecisionTreeClassifier(criterion='gini', 
                                        splitter='best', 
                                        max_depth=4,
                                        class_weight='balanced',
                                        min_samples_leaf=0.1,
                                        random_state=seed_val)
    
    rf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

    rf.fit(X_train, y_train) 
    
    # Evaluate the test set predictions
    pred_y_train = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    
    
    y_pred_proba = rf.predict_proba(X_test)[:,1]
    
    # Calculate accuracy score
    # Create a pd.Series of features importances
    importances = pd.Series(data=rf.feature_importances_,
                            index= X_train.columns)
    # Sort importances
    importances_sorted = importances.sort_values()
    # Draw a horizontal barplot of importances_sorted
    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Features Importances')
    plt.show()
    
    print('*************** Evaluation on Training Data ***************')
    print('Accuracy Score: ', accuracy_score(y_train, pred_y_train))
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_y_train))
    print('--------------------------------------------------------')
    print('------------------Confusion Matrix----------------------')
    print(confusion_matrix(y_train, pred_y_train))
    
    
    print('*************** Evaluation on Test Data ***************')
    print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    # Look at classification report to evaluate the model
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------')
    print(confusion_matrix(y_test, y_pred))
    print('--------------------------------------------------------')    
    print("")
    
    
    print('*************** ROC AUC Score ***************')
    rf_roc_auc = roc_auc_score(y_test,y_pred_proba)
    print("ROC AUC Score {:.3f}".format(rf_roc_auc))
    
    
    


# Select data for modeling

                    
# TestingParameters = ["BMI", "Diabetes", "VitD",\
#                       "Thyroid_Disease", "Drug", "med_calcium carbonate",\
#                           "med_prednisone", "med_lisinopril", "med_lidocaine",\
#                               "med_levothyroxine", "med_fluticasone propionate",\
#                                     "med_aspirin", "cd_dm2"]
TestingParameters = ["Age","Gender","BMI", "Chronic_Diseases","Medication_Risk","Drug_Risk",\
                      "Fracture_Zone","Treatment"] #"Displacement"
                     # "med_prednisone", "med_lisinopril", "med_lidocaine",\
                     #          "med_levothyroxine", "med_fluticasone propionate"]#"cd_dm2"]
    
    
    
sel_meds = ["med_prednisone", "med_lisinopril", "med_lidocaine",\
                              "med_levothyroxine", "med_fluticasone propionate"]
    
taking_meds_list = new_data.loc[:,sel_meds].sum(axis = 1)
taking_meds_list[taking_meds_list>0] = 1
new_data['Medication_Risk'] = taking_meds_list

taking_drug_list = new_data.loc[:,['Drug','med_calcium carbonate']].sum(axis = 1)
taking_drug_list[taking_drug_list>0] = 1
new_data['Drug_Risk'] = taking_drug_list

taking_cd_list = new_data.loc[:,Chronic_Diseases+["Diabetes","Thyroid_Disease"]].sum(axis = 1)
taking_cd_list[taking_cd_list>0] = 1
new_data['Chronic_Diseases'] = taking_cd_list

X = new_data.loc[:,TestingParameters]#['Displacement', 'Fracture_Zone', 'Treatment',
                  
# X = new_data.loc[:,predictor_vars]


# eda_data
y = eda_data[Target_Name].values



#########
# print('Ensemble Training:')
# ensemble_fitting(X, y)

print('RF Training:')
rf_fitting(X,y)

#%%

# Fit the model and display results
X_train, X_test, y_train, y_test, clf, graph = fitting(X, y, 'gini', 'best', 
                                                       mdepth = 4, 
                                                       clweight= 'balanced',
                                                       minleaf = 0.05)

# Plot the tree graph
graph
# tree.plot_tree(clf) 

# Save tree graph to a PDF
#graph.render('Decision_Tree_all_vars_gini')

# fig = Plot_3D(X, X_test, y_test, clf, x1='WindGustSpeed', x2='Humidity3pm', mesh_size=1, margin=1)



#%%




#%%




# #%% Plot features associations
# # eda_data.drop(columns=outcome_cont_vars, inplace=True)
# # eda_data.drop(columns=predictor_cont_vars, inplace=True)
# # eda_data.drop(columns=['HBA1C_Level','Hb_Level','WBC_Level'], inplace=True)

# def cramers_v(x, y):
#     conf_matrix = pd.crosstab(x,y)
#     chi2 = stats.chi2_contingency(conf_matrix)[0]
#     n = conf_matrix.sum().sum()
#     phi2 = chi2/n
#     r,k = conf_matrix.shape
#     phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
#     rcorr = r-((r-1)**2)/(n-1)
#     kcorr = k-((k-1)**2)/(n-1)
#     if (min((kcorr-1),(rcorr-1)) == 0) or phi2corr == 0:
#         return 0
#     else:
#         return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# # from dython.nominal import associations
# # A = associations(eda_data, figsize=(50, 50))

# corr_val = {}
# # keys = range(4)
# # values = ["Hi", "I", "am", "John"]
# # for i in keys:
# #         dicts[i] = values[i]
        

# for cat_id, cat_name in enumerate(TestingParameters):
#     corr_val[cat_name] = cramers_v(eda_data['Healing_Status'], eda_data[cat_name])



# #%%
# # import phik

# # from phik import resources
# # from phik.binning import bin_data
# # from phik.report import plot_correlation_matrix

# # phik_overview = new_data.phik_matrix()

# # plot_correlation_matrix(phik_overview.values, x_labels=phik_overview.columns, y_labels=phik_overview.index, 
# #                         vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=1.5,
# #                         figsize=(15,15))
# # plt.tight_layout()


# # significance_overview = new_data.significance_matrix()





# #%% Statistical Analysis




# #%% Statistical Analysis


# data_binned, binning_dict = bin_data(new_data, retbins=True)


# # plot each variable pair
# plt.rc('text', usetex=False)

# n=0
# for i in range(len(TestingParameters)):
#     n=n+i

# ncols=3
# nrows=int(np.ceil(n/ncols))
# fig, axes = plt.subplots(nrows, ncols, figsize=(15,4*nrows))
# ndecimals = 0

# for i, comb in enumerate(combinations(TestingParameters, 2)):
    
#     c = int(i%ncols)
#     r = int((i-c)/ncols )

#     # get data
#     c0, c1 = comb
#     datahist = data_binned.groupby([c0,c1])[c0].count().to_frame().unstack().fillna(0)
#     datahist.columns = datahist.columns.droplevel()
    
#     # plot data
#     img = axes[r][c].pcolormesh(datahist.values, edgecolor='w', linewidth=1)
    
#     # axis ticks and tick labels
#     if c0 in binning_dict.keys():
#         ylabels = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c0][i][0], binning_dict[c0][i][1])
#                    for i in range(len(binning_dict[c0]))]
#     else:
#         ylabels = datahist.index

#     if c1 in binning_dict.keys():        
#         xlabels = ['{1:.{0}f}_{2:.{0}f}'.format(ndecimals, binning_dict[c1][i][0], binning_dict[c1][i][1])
#                     for i in range(len(binning_dict[c1]))]
#     else:
#         xlabels = datahist.columns
    
#     # axis labels
#     axes[r][c].set_yticks(np.arange(len(ylabels)) + 0.5)
#     axes[r][c].set_xticks(np.arange(len(xlabels)) + 0.5)
#     axes[r][c].set_xticklabels(xlabels, rotation='vertical')
#     axes[r][c].set_yticklabels(ylabels, rotation='horizontal')    
#     axes[r][c].set_xlabel(datahist.columns.name)
#     axes[r][c].set_ylabel(datahist.index.name)    
#     axes[r][c].set_title('data')
    
# plt.tight_layout()













