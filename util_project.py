# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:42:32 2021

This code has all customized functions that we need to clean up datasets

@author: Bardiya
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from dateutil.parser import parse
from sklearn.feature_extraction.text import CountVectorizer
    

    
#%% Data Cleaner Utilities
def data_unifier(df, col_name, col_map = None, df_types = None, show_res = True, zero_out = False):
    '''
    This function is an inplace mapping function to unify the feature values of the dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        This is the dataframe that contains all the features for all our subjects
    col_name : str
        The name of the feature/column needed to be analyzed
    col_map : str
        Mapping the typos or acquired data to unique numbers or values.
        The format has to be {OLD_NAME: NEW_NAME}.
        To avoid mapping, use '' as input.
    col_type : str, optional
        The feature/column data type. It can be 'categorical', 'numerical', 'string'.
    show_res : Boolean, optional
        Print the input and mapped results. The default is True.
    zero_out : Boolean, optional
        Only for categorical variables we need to start from 0 class. The default is False.

    Returns
    -------
    df : pandas.DataFrame
        This is the cleaned dataframe.
        
    '''
    unique_vals = df[col_name].unique()
    
    print('Working on {} column.'.format(col_name))
    
    if show_res:
        print('Raw Data:')
        print(unique_vals)
        
        
    if col_map == None: # this means we're in the data evaluation stage
        print("---> Create your mapping function in the format of {incorrect value/name: corrected value/name}")
        return df 
    else:
        print('Cleaning using the mapping...')
        df.replace({col_name: col_map}, inplace = True)  
        
    if zero_out:
        try:
            print('Zeroing out the values by {}'.format(min(df[col_name])))
            df[col_name] -= min(df[col_name])
        except:
            raise ValueError('There is a string in this column that you did not clean. These are the unique values: {}.'.format(df[col_name].unique()))
        
        
    if df_types is not None:
        print('Data type is {}'.format(df_types[col_name]))
        if df_types[col_name] == 'categorical':
            df[col_name] = df[col_name].astype('category')  
        else:
            print('This type is already set in Pandas...')
            
        
            
    if show_res:
        print('Processed Data (Type: {}):'.format(df[col_name].dtype))
        print(df[col_name].unique())       
        print('-------------------------------')
    
    
    return df



#%% Utilities for string editing
def remove_paranthesis(col_value):
    if type(col_value) == str:
        value_before_p = col_value.split('(')[0]
        value_before_p = value_before_p.replace(',', '.')
        value_before_p = value_before_p.replace('<', '')
        value_before_p = value_before_p.replace('>', '')

        value_before_p = float(value_before_p.strip())
    else:
        value_before_p = col_value
        
    return value_before_p


#### 
def remove_date_after_value(col_value):
    first_number_list = re.findall('^\d+[\.|\,]*\s*\d+', col_value)
    if len(first_number_list)==1: #make sure we have only one number found
        fixed_1 = re.sub(r'(?=\d+)?(\s+)(?=\d+)' , r'', first_number_list[0]) # remove spaces between numbers
        fixed_number = re.sub(r'(?=\d+)?(\,)(?=\d+)'  , r'.', fixed_1) # convert comma to dot between numbers
        try:
            cleaned_value = float(fixed_number)
            # print(col_value, '-->', cleaned_value)
        except:
            print('----- COULD NOT CONVERT THIS:', fixed_number[0])
    else:
        cleaned_value = np.nan

    return cleaned_value

        
####
def isdate(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def get_value_and_date(val_text):
    val_name = np.nan
    val_date = np.nan
        
    if type(val_text) == str:
        val_text = val_text.replace("<","")
        val_text = val_text.replace(">","")
        split_text = re.split(r'[\(\) ]',val_text)
        
        print(split_text)
        for ii, item in enumerate(split_text):
            
            if len(item)>0:
                item = item.strip()

                if not item.isalpha() and ii < 1:
                    val_name = float(item)   
                    
                if item == 'ug/FEU' or item == 'UG/ML':
                    val_name *= 1000
                    
                if ii>1 and isdate(item):
                    val_date = item

    return val_name, val_date


def sep_value_date(column):
    '''
    Parameters
    ----------
    column : string
        The column that is string and has a date in it.

    Returns
    -------
    value : TYPE
        this is the value before the date.
    date : TYPE
        the date that is written.

    '''
    split_data = re.findall(r"(\d+(\.\d+)?)", column)

    try:
        # if len(split_data) !=0:
        if len(split_data) != 3: # this means the value is not written and it's just a date there
            value = split_data[0][0]
        else:
            value = np.nan
        
        try:
            date_complete = [split_data[1][0],split_data[2][0],split_data[3][0]]
            date = "/".join(date_complete)
            
        except:
            date = np.nan
            
        print(column, '-->', value, '---', date)
            
    except:
        value = np.nan
        date = np.nan
        
    return value, date

#%% To find common terms in a text-based format (a, b, c, ..)
# #%% Tokenization of the Medication ... In this section we want to make a term 
# # document matrix or DTM, in our dataset the medications are seperated by ","
# # so we define a new function to split the medication lists by "," and then use
# # the CountVectorizer from SciKit learn to make the DTM matrix... note that 
# # without the tokenize matrix to split the medication with "," the 
# # CountVectorizer would only seperate the words... and since some ofthe 
# # medications include more than one word then our DTM matrix would not be 
# # correct


# # There are so many different medications and our database is not too large so 
# # we need to only take a few of the most popular medications so lets sort out the 
# # columns by the number of their occurance in the data set and pick the most used ones
# # There are different ways to do this.. but here I put the sume of the rows in each column 
# # in a numpy array... then sort the numbers and take the 20 highest ones
# # to make this easier I wrote a function so I can use it for other columns such 
# # as chronic disease too

def check_dictionary(text_array, dict_file):
    '''
    Parameters
    ----------
    text_array : str[] --> this has to be a list, if you have one input, put them as ['TEXT']
        this is an array with each term having its own row
    dict_file : dict()
        this is a dicrionary generated from the .csv file of reference dictionaries
        example:
            pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True).to_dict()

    Returns
    -------
    text_array_unified : str[]
        this is the corrected terms

    '''
    # empty array to compile the full list
    text_array_unified = []
    for text in text_array:
        if type(text) == str: # this is to check if the text is np.nan or not
            value = dict_file.get(text, text) # from the key, read the corrected term
            if value == '':
                text_array_unified.append(text.strip().lower())
            elif value is None:
                text_array_unified.append('none')
            else:
                text_array_unified.append(value.strip().lower())
                
        else: # it means the term is np.nan
            text_array_unified.append('none')
            
                
    return text_array_unified


# Tokenizer for the countVectorizer
def text_preprocessor(text_in):
    text = re.sub('\n', '',text_in)
    text = re.sub('\[|\]', '', text)
    text = text.strip()
    text = re.split(r"[,\/]",text)
    
    tokenized_text = []
    for t in text:
        if t != '':
            if t == 'x' or t == '.':
                tokenized_text.append('none')
            else:
                tokenized_text.append(t.strip('.!:').strip())
    
    return tokenized_text


def med_tokenizer(text_in):
    tokenized_text = text_preprocessor(text_in)
    # path_test = '../../DL Projects/DL_Codes/'
    dictionary_file = 'ref_documents/dictionary_med.csv'
    dict_file = pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True, na_filter= False, usecols=range(2)).to_dict()
    # print(dict_file)
    tokenized_text_out = check_dictionary(tokenized_text, dict_file)
            
    return tokenized_text_out
    
def cd_tokenizer(text_in):
    tokenized_text = text_preprocessor(text_in)
    # path_test = '../../DL Projects/DL_Codes/'
    dictionary_file = 'ref_documents/dictionary_cd.csv'
    dict_file = pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True, na_filter= False, usecols=range(2)).to_dict()
    
    tokenized_text_out = check_dictionary(tokenized_text, dict_file)
            
    return tokenized_text_out

def gd_tokenizer(text_in):
    tokenized_text = text_preprocessor(text_in)
    dictionary_file = 'ref_documents/dictionary_gd.csv'
    dict_file = pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True, na_filter= False, usecols=range(2)).to_dict()
    
    tokenized_text_out = check_dictionary(tokenized_text, dict_file)
            
    return tokenized_text_out    


def findTopItems(df_Column, pre_text = "", just_checking=False):
    """ 
    This function gets a column of a dataframe and returns the Document Term 
    Matrix, with N number of top word occurances
    """
    missed = []
    docs = []
    for c, cell_text in enumerate(df_Column):
        if type(cell_text) == str:
            cell_text = re.sub('\([^)]*\) +','', cell_text)
            cell_text = re.sub('\(|\)', '', cell_text)
            cell_text = re.sub('\-|\+|\%','', cell_text)
            cell_text = re.sub('  +','', cell_text)
            if (cell_text.lower() =='n/a') or (cell_text ==''):
                docs.append('none')
                missed.append(c)
            else:
                med_text = cell_text.strip().lower()
                # if med_text == 'a':
                #     print(cell_text)
                docs.append(med_text)

        else:
            docs.append('none')
            #missed.append(c) # we are saving the row index of the missing data
            # otherwise we would just have 0s for all the missing data
    
    # ### Check Dictionary
    if pre_text == 'med' or pre_text == 'prphlyx':
        vec = CountVectorizer(tokenizer = med_tokenizer)#med_tokenizer)
    elif pre_text == 'cd' or pre_text == 'past_mh' or pre_text == 'cm':
        vec = CountVectorizer(tokenizer = cd_tokenizer) #tokenize)
    elif pre_text == 'gd':
        vec = CountVectorizer(tokenizer = gd_tokenizer) #tokenize)        
    else:
        vec = CountVectorizer(tokenizer = text_preprocessor) #tokenize)
        
    X = vec.fit_transform(docs)
    
    if pre_text == '' or just_checking==True:
        col_names = [cell_text for cell_text in vec.get_feature_names()]
    else:
        col_names = [pre_text + '_' + cell_text for cell_text in vec.get_feature_names()]
    
    
    return col_names, X, vec, missed



def DTM_Maker(df_Column, pre_text='xxx', N=-1, clinician_file = ''):
    """ 
    This function gets a column of a dataframe and returns the Document Term 
    Matrix, with N number of top word occurances
    """
    col_names, X, vec, missed = findTopItems(df_Column, pre_text)
    
    # creating a full matrix with all data points
    DTM = pd.DataFrame(X.toarray(), columns=col_names)

    Occurance = np.array(DTM.sum())
    Prevalence = np.argsort(Occurance) # a has the sorted index of the Occurance.. its ascending  

    if clinician_file == '': #if the file doesn't exist, we return the N top choices
        if N == -1:
            print('Getting all outcomes...')
            TopDTM = DTM
        else:
            TopDTM = DTM.iloc[:, Prevalence[-N:]]
        
    else: # Compare this to the list coming from the clinicians
        clinician_list_in = pd.read_excel(clinician_file)
        clinician_list = clinician_list_in.loc[clinician_list_in.rate>=3,'name']
        clinician_list = pre_text+'_'+clinician_list.values
        clinician_list = [key.strip() for key in clinician_list]
        # print(clinician_list)
        TopDTM = DTM[clinician_list]
        
    #here I am replacing the missing data with np.nan
    for meds in TopDTM.keys():
        TopDTM[meds][missed] = np.nan

    # just in case we are combining terms in the dictionary
    TopDTM[TopDTM>0] = 1
    
    # Visualise the N most common words
    plot_N_most_common_words(X, vec, N, pre_text)
        
    return TopDTM#pd.Series(col_names)



# Helper function
def plot_N_most_common_words(count_data, count_vectorizer, N=10, pre_text='items'):
    title_text = get_title_name_from_text(pre_text)
        
    if N > 25 or N == -1:
        N = 50
        print('Cannot visualize more than 25 items.')
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:N]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    y_pos = np.arange(N)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=600)

    # plt.figure(figsize=(12,12), dpi=300)
    percent_vals = [100* number / count_data.shape[0] for number in counts]
    ax.barh([w.capitalize() for w in words], percent_vals, align='center')    
    # plt.bar(x_pos, counts,align='center')
    ax.invert_yaxis()
    ax.set_ylabel('')
    ax.set_xlabel('% Percent of Patients')

    # plt.ylabel('count')    
    lim_value = round_25(max(percent_vals))+25
    if lim_value > 100:
        lim_value = 100
    ax.set_xlim([0,lim_value])
    ax.set_title('Most Prevalent {}'.format(title_text))
    plt.show()


def round_25(value, base=25):
    rounded_val = base * round(value/base)
    return rounded_val
    
    
#%%% This is Advanced Categorization
def get_title_name_from_text(pre_text='items'):
    if pre_text =='med':
        title_text = 'Medications'
    elif pre_text == 'cd':
        title_text = 'Chronic Diseases'
    elif pre_text == 'gd':
        title_text = 'Genetic Diseases'
    elif pre_text == 'past_mh':
        title_text = 'Medical History'
    elif pre_text == 'prphlyx':
        title_text = 'Prophylaxy Method'  
    elif pre_text == 'cm':
        title_text = 'Comorbidities'          
    else:
        title_text = 'items'
    
    return title_text
    
def medication_categorizer(medication_df, badnames_list, good_name, pre_text='med_'):
    rename_list_in_df = [pre_text+med for med in badnames_list if pre_text+med in medication_df.keys()]
        
    rename_df = medication_df[rename_list_in_df]
    rename_list = rename_df.sum(axis=1)
    rename_list[rename_list>0] =1
    print('Renaming {}\n\t\----> {}'.format(rename_list_in_df, good_name))

    if good_name in badnames_list:
        
        new_df = rename_list + medication_df[pre_text+good_name]
        new_df[new_df>0] = 1
        
        medication_df[pre_text+good_name] = new_df

        keys_not_good = [key for key in rename_list_in_df if pre_text+good_name != key]
        print('Dropping these {}...\n'.format(keys_not_good))
        medication_df.drop(keys_not_good, axis=1, inplace=True)
        
    else:
        medication_df[pre_text+good_name] = rename_list
        print('Dropping these {}...\n'.format(rename_list_in_df))
        medication_df.drop(rename_df, axis=1, inplace=True)
    
    return medication_df


def column_combiner(df1_column, df2_column):
    print('Before Combining, we have {} values in first df, and {} values in the second df.'.format(df1_column.sum(), df2_column.sum()))

    Temp_1 = df1_column.fillna(0)
    Temp_2 = df2_column.fillna(0)
    df2_column = Temp_1 + Temp_2
    df2_column[df2_column>0] = 1
    
    print('-After Combining, we have {} values in the second df.'.format(df2_column.sum()))

    return df2_column


def get_N_common_df(df_curated, N = 25, pre_text=''):
    n_subjects = len(df_curated)
    Occurance = np.array(df_curated.sum())
    Occurance_sorted = np.sort(Occurance) 
    Prevalence = np.argsort(Occurance) 
    
    n_occurance = sorted(Occurance)
    n_occurance = n_occurance[-N:]
    
    TopDTM = df_curated.iloc[:, Prevalence[-N:]]
    
    count_dict = (zip(TopDTM.keys(), n_occurance))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)
        
    trim_text = len(pre_text)
    words = [w[0][trim_text:] for w in count_dict]
    
    return TopDTM, pd.DataFrame({'item':[word for word in words], 'occurance': [num for num in Occurance_sorted[:-N-1:-1]]})
    

def plot_N_common(df_curated, N=25, pre_text='items'):

    title_text = get_title_name_from_text(pre_text)
        
    n_subjects = len(df_curated)
    
    Occurance = np.array(df_curated.sum())
    Prevalence = np.argsort(Occurance) 
    
    n_occurance = sorted(Occurance)
    n_occurance = n_occurance[-N:]
    
    TopDTM = df_curated.iloc[:, Prevalence[-N:]]
    
    count_dict = (zip(TopDTM.keys(), n_occurance))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)
        
    trim_text = len(pre_text)+1
    words = [w[0][trim_text:] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    y_pos = np.arange(N)
    fig, ax = plt.subplots(1, 1, figsize=(12,12), dpi=1200)
    
    # plt.figure(figsize=(12,12), dpi=300)
    percent_vals = [100* number / n_subjects for number in counts]
    ax.barh([w.upper() for w in words], percent_vals, align='center')    
    # plt.bar(x_pos, counts,align='center')
    ax.invert_yaxis()
    ax.set_ylabel('')
    ax.set_xlabel('% Percent of Patients')
    
    # plt.ylabel('count')    
    lim_value = round_25(max(percent_vals))+25
    if lim_value > 100:
        lim_value = 100
    ax.set_xlim([0,lim_value])
    ax.set_title('Most Prevalent {}'.format(title_text))
    plt.show()



    
   
#%% Laboratory Data Analyzer
def get_lab_dict(text_lab):
    dictionary_file = 'ref_documents/dictionary_lab.csv'
    dictionary_of_lab_names = pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True, na_filter= False, usecols=range(2)).to_dict()

    lab_dict = {}
    if type(text_lab) == str:
        this_row = text_lab.rstrip().lstrip().rstrip(',').lstrip(',')
        
        this_row = re.sub(r'\([a-zA-Z]*\):' , '', this_row) ## remove something like "(PE):" or these type of things
            
        this_row = re.sub(r'^\([^)]*\)', '', this_row) ## remove the date if it's written at the begining of the sentence

        this_row = re.sub(r'(?=\d+)?(\,)(?=\d+)' , r'.', this_row) ## replace the comma between nnumbers to a dot
            
        split_list = this_row.lower().strip().split(',')

        for item in split_list:
            if len(item)>0:
                ## check if it's "x"
                if (item == 'x') or (item == 'no labs') or (item == 'normal labs'):
                    lab_dict['normal_labs'] = float(1)
                    break
                    
                # clean-up the item name
                item = lab_regexp(item)
                    
                # ## check if it's nolabs entry
                # if item == 'nolabs':
                #     lab_dict['lab_normal_y'] = float(1)
                #     break
                    
                ## try to separate them
                try:
                    lab_name, lab_val = item.split('=')
        
                    lab_name = check_dictionary([lab_name], dictionary_of_lab_names)
                    
                    if lab_val == "notavailable":
                        lab_dict[lab_name[0]] = float(np.nan)
                    elif lab_val.lower() == 'positive':
                        lab_dict[lab_name[0]] = 1
                    else:
                        lab_dict[lab_name[0]] = float(lab_val)
                    
                except:
                    print(this_row, ' -----CHECK:    ', lab_name, lab_val)
                    raise ValueError('Could not extract the lab data... Check the "find_bad_lab_data" function"...')
                                     
            else: # if there is no entry for this patient
                lab_dict['normal_labs'] = float(1)
                       
        
    elif len(lab_dict) == 0:
        lab_dict['normal_labs'] = float(1)
            
        
    return lab_dict



def lab_regexp(item):
    # item = item.replace("not available", "")
    # item = item.replace("not applicable", "")
    item = item.replace(" ", "")
    item = item.replace("<", "=")
    item = item.replace(">", "=")
    item = item.replace("=+", "=")
    item = item.replace(",,",",")
    item = item.strip('.!,%')
    
    return item


def find_bad_lab_data(df, col_name):
    dictionary_file = 'ref_documents/dictionary_lab.csv'
    dictionary_of_lab_names = pd.read_csv(dictionary_file, header=None, index_col=0, squeeze=True, na_filter= False, usecols=range(2)).to_dict()


    print('Fix these lab data for {} column'.format(col_name))
    lab_list = set()
    for i in range(len(df)):
        this_row = df[col_name][i]
        if type(this_row) == str:

            this_row = this_row.rstrip().lstrip().rstrip(',').lstrip(',')

            this_row = re.sub(r'\([a-zA-Z]*\):' , '', this_row) ## remove something like "(PE):" or these type of things
            
            this_row = re.sub(r'^\([^)]*\)', '', this_row) ## remove the date if it's written at the begining of the sentence

            this_row = re.sub(r'(?=\d+)?(\,)(?=\d+)' , r'.', this_row) ## replace the comma between nnumbers to a dot
            
            split_list = this_row.lower().split(',')
    
            for item in split_list:
                
                if len(item)>0:
                    
                    if (item == 'x') or (item == 'no labs'):
                        lab_name = 'lab_normal_y'
                        lab_val = 1
                        lab_list.add(lab_name)
                        break
                    
                    item = lab_regexp(item)
                    
                    
                    try:
                        lab_name, lab_val = item.split('=')
                        
                        lab_name = check_dictionary([lab_name], dictionary_of_lab_names)[0]
                        
                        
                        if lab_val == "notavailable":
                            lab_val = float(np.nan)
                        elif lab_val.lower() == 'positive':
                            lab_val = 1
                        else:
                            lab_val = float(lab_val)
                        
                        lab_list.add(lab_name)

                    except:
                        print(df['MRN/EMPI'][i], this_row, '\n\t\t\t\t----> CHECK:\t', item.strip())
                        
                        
                else:
                    lab_name = 'lab_normal_y'
                    lab_val = 1
                    lab_list.add(lab_name)
                    
                
                            

    print('Finished checking...\n')    
    
    return lab_list


# #%% FIX IF NORMAL LAB DATA:
# def assume_lab_normal(lab_column_as_dict):
#     output_columns = pd.Series(lab_column_as_dict)
#     print('Not defined yet...')
#     return output_columns
    

#%%    

def get_cat_data(path_description_file, categorical_vars, excluding_columns):
 
    # Load Description file
    description_file = pd.read_excel(path_description_file)   
        
    # Assign category keys to each variable
    cat_vars_bool = (description_file.type == 'Categorical') & (description_file.drop_list != 'y')\
                        & (description_file.new_name != excluding_columns)
    cat_list_temp  = description_file.new_name[cat_vars_bool]
    cat_ticks_temp = description_file.cat_meaning[cat_vars_bool]

    cat_list = []
    cat_ticks = []
    for cat_name, ticks_list in zip(cat_list_temp, cat_ticks_temp):
        cat_list.append(cat_name.strip().encode().decode('utf_8_sig'))
        cat_ticks.append(ticks_list.strip().encode().decode('utf_8_sig'))
    
    
    print('\nCategorical Variables are:')
    for cn, ct in zip(cat_list,cat_ticks):
        print('{:20}:\t{}'.format(cn, ct))
    
    
    cat_ticks = [cat_ticks[i] for i, cat_name in enumerate(cat_list) if cat_name in categorical_vars]
    cat_list = [cat_name for i, cat_name in enumerate(cat_list) if cat_name in categorical_vars]


    cat_dic = dict(zip(cat_list, cat_ticks))


    cat_len_temp = [len(n.split(', ')) for n in cat_ticks]
    cat_len = dict(zip(cat_list, cat_len_temp))

    return cat_dic, cat_len    






#%% Visualizations
def nan_plot(dataframe, print_top_10 = True, visualize = True, title_text = ''):
    NA_number = np.array(dataframe.isna().
                      sum().
                      reset_index(name = 'na'))
    NA_percent = 100*NA_number[:,1]/len(dataframe)
    
    # We need to know how much in % we are am missing...
    if visualize:
        plt.figure(dpi = 150, figsize=(10, 6))
        plt.bar(NA_number[:,0],NA_percent, align='center', width=0.7)
        plt.gcf().subplots_adjust(bottom=0.2) # to make room for the x axis labels 
        plt.xticks(rotation=90,fontsize = 5)
        plt.yticks(rotation=90,fontsize = 8)
        plt.ylabel('% missing items')
        plt.title(title_text)
        
    missing_vals = dataframe.isnull().mean().sort_values(ascending = False)*100
    if print_top_10:    print(missing_vals[missing_vals>10])
    
    return dataframe.isnull().mean()*100