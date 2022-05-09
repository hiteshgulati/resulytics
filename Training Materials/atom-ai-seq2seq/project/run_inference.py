'''
Script to run inference of saved seq2seq model on given source file. Takes single config JSON file as input.
'''
import sys
import json
import pandas as pd
import torch
import inference
import utils
import os
import numpy as np
import prepdata

if len(sys.argv) != 2:
    raise ValueError('Please provide single config JSON file as input')

input = sys.argv[1]
extension = input.split('.')[-1]

if extension != 'json':
    raise ValueError('Please provide single config_infer JSON file as input')

with open(sys.argv[1], 'r') as f:
    config = json.load(f)

if(torch.cuda.is_available()): device = torch.device("cuda")
else: device = torch.device("cpu")

path = config['inference']['file']['path']
files = config['inference']['file']['files']
sheets = config['inference']['file']['sheets']
headers = config['inference']['file']['headers']
verbose = config['inference']['write']['verbose']
file_format = config['file_format']

file_extension = utils.check_file_types(files)
file_name = "_".join([".".join(file.split('.')[:-1]) for file in files])

write_csv = config['inference']['write']['write_predictions_csv']
write_path = config['inference']['write']['write_csv_path']
if not os.path.exists(write_path) and write_path != None: os.makedirs(write_path)

#load abbreviations file for preprocessing
abbrev_file = config['training']['preprocess']['abbreviations_file']
abbreviations_file = pd.read_csv(abbrev_file, dtype=str, header=0)

#load saved model
model_path = config['inference']['read']['save_model_path']
model_name = config['inference']['read']['model_name']

input_set, output_set, encoder, decoder, max_length, model_type = utils.load_model(model_path, model_name, device)

if(verbose): print(f'Model {model_name} successfully loaded')

#preprocess data
df, have_targets = prepdata.create_merged_df(path, files, headers, sheets, file_extension, file_format) 
if(verbose): print(f'preprocessing data ({len(df)} accounts)... ', end=" ", flush=True)
df = prepdata.preprocess_df(df, model_type, have_targets, file_format, abbreviations_file)
if(verbose): print('done')

#perform inference on given dataset and output predictions with confidence scores
df_predictions = inference.predict_on_unknown(input_set, output_set, encoder, decoder, model_type, df, file_format, max_length=max_length, verbose=verbose)
#if targets included in input, append ground truth targets with calculated bleu scores
if have_targets:
    df_predictions = inference.evaluate_output(df_predictions, df, model_type, file_format)

#format output file
df_predictions = utils.format_to_upper(df_predictions)
    
if(verbose): print(df_predictions.head(5))

write_test = config['inference']['write']
if(write_test['write_predictions_csv'] == True): 
    write_pred_path = write_test['write_csv_path']
    df_predictions.to_csv(write_pred_path+file_name+'_predictions.csv', index=False)
    if(verbose): print(f'test predictions written to {write_pred_path}')