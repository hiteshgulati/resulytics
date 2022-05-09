'''
Script to train and evaluate seq2seq model on given list of files containing source and target accounts. Takes single config JSON file as input.
'''
import sys
import os
import json
import torch
import pandas as pd
import prepdata
import word_encoding
import seq2seq_model
import train
import inference
import utils
from sklearn.model_selection import train_test_split

if len(sys.argv) != 2:
    raise ValueError('Please provide single config JSON file as input')

input = sys.argv[1]
extension = input.split('.')[-1]

if extension != 'json':
    raise ValueError('Please provide single config JSON file as input')

with open(sys.argv[1], 'r') as f:
    config = json.load(f)
    
#input file params
file_info = config['training']['file']
data_path = file_info['path']
file_list = file_info['files']
header_list = file_info['headers']
sheet_list = file_info['sheets']
file_format = config['file_format']

#preprocess data params
preproc = config['training']['preprocess']
shuffle = preproc['shuffle_data']
test_size = preproc['test_size']
model_type = preproc['model_type']
abbrev_file = preproc['abbreviations_file']

#save train and test file params
verbose = config['training']['write']['verbose']
save_files_info = config['training']['write']['save_train_and_test']
save_files = save_files_info['save_files']
write_files_path = save_files_info['save_to']
train_file_name = save_files_info['train_file_name']
test_file_name = save_files_info['test_file_name']

#validation
val_info = config['training']['validation']
validate = val_info['validate']
if(validate): 
    val_file = val_info['file_path'] + val_info['file_name']
    val_df = utils.load_df(val_file, val_info['sheet'], val_info['header'])
else: 
    val_df = None
    
#model params
save_model = config['training']['write']['save_model']
save_model_path = save_model['save_to']
model_name = save_model['model_name']
save_every = save_model['save_every']
params = config['training']['model_params']

if(torch.cuda.is_available()): 
     if(verbose): print("model running on GPU")
     device = torch.device("cuda")
else: 
     if(verbose): print("model running on CPU")
     device = torch.device("cpu")

if(test_size == 0): testing = False
else: testing = True
        
abbreviations_file = pd.read_csv(abbrev_file, dtype=str, header=0)
    
#format data for model
x_train, x_test, y_train, y_test = prepdata.data_preprocessing_pipeline(data_path, file_list, header_list, sheet_list, abbreviations_file, model_type, file_format, test_size=test_size, verbose=verbose)

if(save_files) == True: 
    file_type = utils.check_file_types(file_list)
    prepdata.save_train_and_test_files(x_train, x_test, y_train, y_test, file_type, write_path=write_files_path, 
                                       train_file_name=train_file_name, test_file_name=test_file_name, verbose=verbose)

input_set, output_set, pairs = word_encoding.create_vocabulary('legacy', 'new', x_train, y_train)

if(verbose): print(f'\ninitializing model with parameters:\n{params}')
    
#initialize model
encoder = seq2seq_model.Encoder(input_set.n_words, params['hidden_size']).to(device)

attention = params['attention']['use_attention']
if attention == True:
    dropout = params['attention']['dropout']
    if(verbose): print(f'Using Attention Decoder with dropout = {dropout}')
    decoder = seq2seq_model.AttnDecoder(params['hidden_size'], output_set.n_words, dropout_p=dropout, max_length=params['max_length']).to(device)
else:
    if(verbose): print('Using Standard Decoder')
    dropout = None
    decoder = seq2seq_model.Decoder(params['hidden_size'], output_set.n_words).to(device) 

#train model on training set
train.train_model(input_set, output_set, encoder, decoder, pairs, n_iters=params['num_iters'], save_path=save_model_path, model_name=model_name, model_type=model_type,
                  hidden_size=params['hidden_size'], validate=validate, validation_data=val_df, abbreviations_file=abbreviations_file, dropout=dropout, attention=attention, 
                  benchmark_every=params['benchmark_loss_every'], save_every=save_every, learning_rate=params['learning_rate'], tf_ratio=params['teacher_forcing_ratio'], 
                  max_length=params['max_length'], verbose=verbose)  


if(testing):
    #evaluate on test set
    data_test = pd.concat([x_test, y_test], axis=1)
    data_test = data_test.reset_index(drop=True)
    data_test['Source Account Number'], data_test['Target Account Number'] = "", ""
    
    df_predictions = inference.predict_on_unknown(input_set, output_set, encoder, decoder, model_type=model_type, df=data_test, max_length=params['max_length'], verbose=verbose)
    df_predictions = inference.evaluate_output(df_predictions, data_test, model_type=model_type)
    
    df_predictions = utils.format_to_upper(df_predictions)
    
    write_test = config['inference']['write']
    if(write_test['write_predictions_csv'] == True): 
        write_pred_path = write_test['write_csv_path']
        df_predictions.to_csv(write_pred_path+'test_predictions.csv', index=False)
        if(verbose): print(f'test predictions written to {write_pred_path}')

    if(verbose): print(df_predictions.head(5))
        