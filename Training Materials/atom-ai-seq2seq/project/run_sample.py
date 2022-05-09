import sys
import pandas as pd
import torch
import inference
import utils
import os
import errno
import numpy as np
import prepdata
import utils
import inference
import json

cwd = os.path.abspath(os.getcwd())
data_path = cwd+os.sep+'data'+os.sep
abbrev_file_path = cwd+os.sep+'ShortForms.csv'
model_path = cwd+os.sep+'model'+os.sep
write_path = cwd+os.sep+'predictions'+os.sep
verbose = True
with open(cwd+os.sep+'config.json', 'r') as f:
    config = json.load(f)
file_format = config['file_format']
print()

if len(sys.argv) == 2:
    file = sys.argv[1]
else:
    raise ValueError('Incorrect number of inputs. Please run script with a single excel file (.xlsx) as input')
    
file_name = ".".join(file.split('.')[:-1])
file_extension = file.split('.')[-1]
if file_extension not in ['xlsx', 'csv']:
    raise TypeError('Invalid file type. Please provide single excel/csv file (.xlsx/.csv) as input')

print(f'Looking for file {file} in {data_path}...')
if not os.path.exists(data_path+file):
     raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
        
if utils.preprocess_text(file_name) == 'example':
    header = 0
    sheet = 'Sheet1'
else:
    header = int(input('Please input file header value: '))
    if file_extension == 'xlsx': sheet = input('Please input file sheet name: ')
    else: sheet = None
        

df, have_targets = prepdata.create_merged_df(data_path, [file], [header], [sheet], file_extension, file_format) 
print(f'File {file} opened successfully')

#examples without hierarchy; short
if file_name in ['Example1', 'Example3']:
    model_type = 'short'
    data_type = 'BS'
#examples with hierarchy; long
elif file_name in ['Example2', 'Example4']:
    model_type = 'long'
    data_type = 'BS'
else:
    data_type = None
    while(data_type not in ['BS', 'IS']):
        data_type = input('Please input given data type (BS/IS): ')
        if data_type not in ['BS', 'IS']: print('Error: invalid input')
        
    model_type = None
    while(model_type not in ['short', 'long']):
        model_type = input('Please input selected model type (short/long): ')
        if model_type not in ['short', 'long']: print('Error: invalid input')

#load abbreviations file for preprocessing
abbreviations_file = pd.read_csv(abbrev_file_path, dtype=str, header=0)

#preprocess data
if(verbose): print(f'preprocessing data ({len(df)} accounts)... ', end=" ", flush=True)
df = prepdata.preprocess_df(df, model_type, have_targets, file_format, abbreviations_file)
if(verbose): print('done')

model_name = utils.get_sample_model_name(data_type, model_type)

#load saved model
if(torch.cuda.is_available()): device = torch.device("cuda")
else: device = torch.device("cpu")

input_set, output_set, encoder, decoder, max_length, model_type = utils.load_model(model_path, model_name, device)
print(f'Model {model_name} successfully loaded')

#reset data indices
df = df.reset_index(drop=True)

#perform inference on given dataset and output predictions with confidence scores
df_predictions = inference.predict_on_unknown(input_set, output_set, encoder, decoder, model_type, df, file_format, max_length=max_length, verbose=verbose)
#if targets included in input, append ground truth targets with calculated bleu scores
if have_targets:
    df_predictions = inference.evaluate_output(df_predictions, df, model_type, file_format)

#format output file
df_predictions = utils.format_to_upper(df_predictions)

df_predictions.to_csv(f"{write_path}{file_name}_predictions.csv", index=False)
if(verbose): print(f'Prediction file written to: {write_path}')