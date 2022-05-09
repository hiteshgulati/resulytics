import utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_merged_df(data_path, file_list, header_list, sheet_list, file_type, file_format):
    
    if len(file_list) != len(header_list):
        raise ValueError('Lengths of file_list and header_list do not match')
        
    df = pd.DataFrame()
    targets_check = []
    
    for f in range(len(file_list)):
        
        file_name = file_list[f]
        header = header_list[f]
        
        if file_type == 'csv':
            df_file = pd.read_csv(data_path+file_name, dtype=str, header=header, encoding='latin-1')
        elif file_type == 'xlsx':
            sheet_name = sheet_list[f]
            df_file = pd.read_excel(data_path+file_name, engine='openpyxl', dtype=str, header=header, sheet_name=sheet_name)
        else: 
            raise ValueError('Invalid input for file_type. Acceptable inputs include: csv, excel')

        if(file_format['Target Acct Description'] in df_file.columns): targets_check.append(1)
        else: targets_check.append(0)
            
        df = pd.concat([df, df_file], sort=False)


    if(len(set(targets_check)) > 1):
        raise ValueError('Inconsistency with inclusion/exclusion of target data between specified files')

    have_targets = set(targets_check).pop()
        
    return df, have_targets


def format_df(df, model_type, have_targets, file_format):

    df = df.dropna(how='all')

    SAN, SL1, SL2, SL3, SAD, TAN, TL1, TL2, TL3, TAD = utils.get_format_names(model_type, have_targets, file_format)
    
    if model_type == 'long':

        if(any([SL1 == None, SL2 == None, SL3 == None])):
            raise ValueError("Model type 'long' selected, but input file source level names not given")

        source_data = pd.DataFrame({SAN: df[SAN].copy(), SL1: df[SL1].copy(), 
                                    SL2: df[SL2].copy(), SL3: df[SL3].copy(), 
                                    SAD: df[SAD].copy()})

        if(have_targets):
            target_data = pd.DataFrame({TAN: df[TAN].copy(), TL1: df[TL1].copy(), 
                                        TL2: df[TL2].copy(), TL3: df[TL3].copy(), 
                                        TAD: df[TAD].copy()})
        else: target_data = None
        
    elif model_type == 'short':
        source_data = pd.DataFrame({SAN: df[SAN].copy(), 
                                    SAD: df[SAD].copy()})
        
        if(have_targets):
            target_data = pd.DataFrame({TAN: df[TAN].copy(), 
                                        TAD: df[TAD].copy()})
        else: target_data = None
        
    else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")

    df = pd.concat([source_data, target_data], axis=1)        
    if model_type == 'long':
        #filter entries with blank levels
        df = df[df[SL1].notnull() & df[SL2].notnull() & df[SL3].notnull()]
        
    return df


def preprocess_df(df, model_type, have_targets, file_format, abbreviations_file):
    
    _, SL1, SL2, SL3, SAD, _, TL1, TL2, TL3, TAD = utils.get_format_names(model_type, have_targets, file_format)

    if model_type == 'long':
            if(any([SL1 == None, SL2 == None, SL3 == None])):
                raise ValueError("Model type 'long' selected, but input file source level names not given")

    abbrev_file = abbreviations_file.copy().dropna() 
    abbrev_file['Abbreviations'] = abbrev_file['Abbreviations'].str.lower()
    
    for col in df.columns:
        if(col not in [SL1, SL2, SL3, SAD, TL1, TL2, TL3, TAD]): continue
        else:
            df[col] = df[col].apply(utils.preprocess_text) 
            df[col] = df[col].apply(utils.find_and_replace_abbr, abbrev_file=abbrev_file)
            df[col] = df[col].apply(utils.preprocess_text) #preprocess replaced abbreviations
            
    return df
    

def prepare_data_for_model(df, model_type, have_targets, file_format, test_size=0.2, shuffle=True):
    
    _, SL1, _, _, SAD, _, TL1, _, _, TAD = utils.get_format_names(model_type, have_targets, file_format)

    if(model_type == 'long'):
        source_data = df.loc[:, SL1:SAD]
        if(have_targets): target_data = df.loc[:, TL1:TAD]
    else:
        source_data = pd.DataFrame({SAD: df[SAD]})
        if(have_targets): target_data = pd.DataFrame({TAD: df[TAD]})

    if(have_targets):
        if(test_size == 0): 
            x_train, y_train = source_data, target_data
            x_test, y_test = None, None 
        else: 
            x_train, x_test, y_train, y_test = train_test_split(source_data, target_data, test_size=test_size, shuffle=shuffle)
    else:
        if(test_size == 0): 
            x_train = source_data
            y_train, x_test, y_test = None, None, None
        else:
            x_train, x_test = train_test_split(source_data, test_size=test_size, shuffle=shuffle)
            y_train, y_test = None, None

    return x_train, x_test, y_train, y_test
    

def data_preprocessing_pipeline(data_path, file_list, header_list, sheet_list, abbreviations_file, model_type, file_format, test_size=0.2, verbose=True):

    if(verbose): print('starting data preprocessing...')

    if(verbose): print('checking given file(s) type... ', end=" ", flush=True)
    file_type = utils.check_file_types(file_list)
    if(verbose): print(f'{file_type} file detected')

    if(verbose): print('merging files into single data frame... ', end=" ", flush=True)
    df, have_targets = create_merged_df(data_path, file_list, header_list, sheet_list, file_type, file_format)
    
    if(verbose): print('formatting data frame... ', end=" ", flush=True)
    df = format_df(df, model_type, have_targets, file_format)

    if(verbose): print('preprocessing data... ', end=" ", flush=True)
    df = preprocess_df(df, model_type, have_targets, file_format, abbreviations_file)
    if(verbose): print(f'done')

    if(verbose): print(f'data frame created with {len(df)} entries')
        
    #df.to_csv(os.getcwd()+'/df_preproc.csv', index=False, mode='w+')
        
    x_train, x_test, y_train, y_test = prepare_data_for_model(df, model_type, have_targets, file_format, test_size=test_size, shuffle=True)

    if(verbose): print(f'training size: {len(x_train)}')

    return x_train, x_test, y_train, y_test

    
def save_train_and_test_files(x_train, x_test, y_train, y_test, file_extension, write_path, train_file_name='data_train', test_file_name='data_test', verbose=True):

    data_train = pd.concat([x_train, y_train], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)
    
    if file_extension == 'csv':
        data_train.to_csv(write_path+f'{train_file_name}.csv', index=False)
        data_test.to_csv(write_path+f'{test_file_name}.csv', index=False)
    elif file_extension == 'xlsx':
        data_train.to_excel(write_path+f'{train_file_name}.xlsx', index=False)
        data_test.to_excel(write_path+f'{test_file_name}.xlsx', index=False)
        
    if(verbose): print(f'Train and test files written to {write_path}')
    

if __name__ == '__main__':
        
    data_path = '/home/ec2-user/SageMaker/project/data/grouped_data/system/'
    file_list = ['SAP.csv', 'Oracle.csv', 'MSDynamics.csv']
    header_list = [0, 0 , 0]
    sheet_list = None
    model_type = 'long'
    test_size = 0.15

    x_train, x_test, y_train, y_test = data_preprocessing_pipeline(data_path, file_list, header_list, sheet_list, model_type, test_size=test_size, verbose=True)
    print(x_train.head(3))

    save_train_and_test_files(x_train, x_test, y_train, y_test, file_extension='csv', write_path='/home/ec2-user/SageMaker/project/data/', 
                              train_file_name='data_train', test_file_name='data_test', verbose=True)


