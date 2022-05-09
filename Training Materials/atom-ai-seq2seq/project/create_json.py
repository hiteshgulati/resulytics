import json
import os

cwd = os.path.abspath(os.getcwd()) + os.sep

config = {
        'file_format':{
                'Source Acct #': 'Source GL Account',
                'Source Acct Description': 'Source GL Account Description',
                'Source Level 1': 'Source L1', 
                'Source Level 2': 'Source L2', 
                'Source Level 3': 'Source L3', 

                'Target Acct #': 'Target GL Account',
                'Target Acct Description': 'Target GL Account Description',
                'Target Level 1': 'Target L1', 
                'Target Level 2': 'Target L2', 
                'Target Level 3': 'Target L3'},

    
        'training':{
                'file':{  
                    'path': cwd+"data/",
                    'files': ["Final Output.csv"],     
                    'sheets': None,  
                    'headers': [0]},
                'preprocess':{ 
                    'shuffle_data': True,
                    'test_size': 0.15,
                    'model_type': 'short',
                    'abbreviations_file': cwd+'ShortForms.csv'},
                'write':{
                    'verbose': True,
                    'save_train_and_test':{
                        'save_files': False,
                        'save_to': None,
                        'train_file_name': None,
                        'test_file_name': None},
                    'save_model':{
                        'save_to': cwd+"model/",
                        'model_name': 'v0.3_long_BS',
                        'save_every': 2000}},
                'model_params':{
                    'hidden_size': 256,
                    'num_iters': 100000,
                    'benchmark_loss_every': 1000,
                    'learning_rate': 0.01,
                    'teacher_forcing_ratio': 0.0,
                    'attention':{
                        'use_attention': True,
                        'dropout': 0.1},
                    'max_length': 100},
               'validation':{
                    'validate': False,
                    'file_path': None,
                    'file_name': None,
                    'sheet': None,
                    'header': None}},
                
        'inference':{
                'file':{  
                    'path': cwd+"data/",
                    'files': ["Final Output.csv"],
                    'sheets': [None],
                    'headers': [0]},
                'read':{
                    'save_model_path': cwd+"model/",
                    'model_name': 'v0.3_long_BS_0.146.hdf5'},
                    #'model_name': 'v0.3_short_BS_1.320.hdf5'},
                'write':{
                    'verbose': True,
                    'write_predictions_csv': True,
                    'write_csv_path': cwd+"predictions"+os.sep}},
}             
                
with open('config.json', 'w') as f:
    json.dump(config, f)
    
print('write successful')
