'''
Helper functions for other modules in seq2seq package. 
'''

from fuzzywuzzy import fuzz
import os
import torch
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
import word_encoding
import seq2seq_model

def get_format_names(model_type, have_targets, file_format):

    #initialize
    SAN, SAD, TAN, TAD = None, None, None, None
    SL1, SL2, SL3, TL1, TL2, TL3 = None, None, None, None, None, None

    SAN, SAD = file_format['Source Acct #'], file_format['Source Acct Description']
    if have_targets: 
        TAN, TAD = file_format['Target Acct #'], file_format['Target Acct Description']

    if model_type == 'long':
        SL1, SL2, SL3 =  file_format['Source Level 1'], file_format['Source Level 2'], file_format['Source Level 3']
        if have_targets: 
            TL1, TL2, TL3 =  file_format['Target Level 1'], file_format['Target Level 2'], file_format['Target Level 3']

    return SAN, SL1, SL2, SL3, SAD, TAN, TL1, TL2, TL3, TAD


def preprocess_text(text):

    text = str(text)
    text = re.sub('-', ' ', text)
    text = re.sub('–', ' ', text)
    text = re.sub("'", '', text)
    text = text.split(' ')
   
    words = []

    for word in text:

        #handle text with forward slash
        fs_split = word.split('/')
        if len(fs_split) > 1:
            if(list(filter(lambda word: len(word) < 3, fs_split))): word = "".join(fs_split)
            else: word = " ".join(fs_split)

        #handle text with ampersand 
        amp_split = word.split('&')
        if len(amp_split) > 1:
            if(list(filter(lambda word: len(word) < 3, amp_split))): word = "".join(amp_split)
            else: word = " ".join(amp_split)

        #handle text with period abbreviations
        per_split = word.split('.')
        if len(per_split) > 1:
            if(list(filter(lambda word: len(word) < 3, per_split))): word = "".join(per_split)
            else: word = " ".join(per_split)
        
        #handle single strings with words separated by capitals
        chars = [char for char in word]

        idx_slices = [0]
        for i in range(len(chars)):
            if i == 0: continue
            #want to keep consecutive letters together
            if chars[i-1] in string.ascii_uppercase: continue
            if chars[i] in string.ascii_uppercase: idx_slices.append(i)

        word = " ".join([word[i:j] for i,j in zip(idx_slices, idx_slices[1:]+[None])])
        words.append(word)
    
    text = " ".join(words).lower().strip()
    text = "".join([char for char in text if char not in string.punctuation+string.digits])
    text = re.sub('\\s+', ' ', text).strip()

    return text


def find_and_replace_abbr(account_descr, abbrev_file):
    
    descr_words = [word for word in account_descr.split() if word != 'nan']

    for i in range(len(descr_words)):
        word = descr_words[i].lower()
        if word in abbrev_file['Abbreviations'].tolist():
            full_word = abbrev_file[abbrev_file['Abbreviations'] == word]['FullForm'].unique()[0]
            descr_words[i] = full_word
            
    return " ".join(descr_words)


def format_to_upper(df):
    
    exclude = ['Source Account Number', 'Confidence Score', 'Target Account Number', 
               'Predicted Level 1 Bleu Score', 'Predicted Level 2 Bleu Score', 'Predicted Level 3 Bleu Score',
               'Predicted Description Bleu Score', 'Predicted Description Similarity']
    
    for col in df.columns:
        if col in exclude: continue
        else: 
            df[col] = df[col].str.upper()
            
    return df


def eliminate_duplicate_words(prediction):
    
    pred_words = []
    for word in prediction.split():
        if word in pred_words: continue
        else: pred_words.append(word)
           
    cleaned_prediction = " ".join(pred_words) 
    
    return cleaned_prediction
    

def check_file_types(file_list):
    
    file_extensions = []
    for file in file_list:
        extension = file.split('.')[-1]
        file_extensions.append(extension)
    
    if len(set(file_extensions)) > 1:
        raise TypeError('Please make sure all specified files are of the same type (csv or excel)')

    file_extension = set(file_extensions).pop()

    if(file_extension not in ['xlsx', 'csv']):
        raise ValueError('Invalid file type. Acceptable files include: csv, excel')
        
    return file_extension


def load_df(file_path, sheet_name, header):
    
    file_extension = check_file_types([file_path])
    if(file_extension == 'csv'):
        df = pd.read_csv(file_path, dtype=str, header=int(header))
    elif(file_extension == 'xlsx'):
        df = pd.read_excel(file_path, engine='openpyxl', dtype=str, header=int(header))
    else:
        raise ValueError('Invalid file type. Acceptable files include: csv, excel')
        
    return df


def plot_loss(model_name, train_losses, val_losses, n_iters, benchmark_every, learning_rate):
    '''
    Plot model loss throughout training. 
    '''
    x = np.arange(0, n_iters, benchmark_every)
    plt.figure()
    plt.plot(x, train_losses, label = 'training')
    if len(val_losses) > 0: plt.plot(x, val_losses, label = 'validation')
    plt.legend()
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    current_dir = os.getcwd()
    save_to = os.path.join(current_dir, r'loss_plots')
    if not os.path.exists(save_to): os.makedirs(save_to)
    plt.savefig(os.path.join(save_to, f'loss_{model_name}_{learning_rate}.jpg'))
    plt.show()
    

def calc_bleu_score(reference, hypothesis):
    reference = [str(reference).split()]
    hypothesis = str(hypothesis).split()
    return round(sentence_bleu(reference, hypothesis, weights=(1,)), 3)
    
    
def calc_scores(predictions, targets, model_type):

    l1_scores, l2_scores, l3_scores, l4_scores = [],[],[],[]
    description_similarities = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        tar = targets[i]
        
        if(model_type == 'short'):
            bleu_score = calc_bleu_score(tar[0], pred)
            description_similarity = fuzz.token_sort_ratio(tar, pred)/100
            description_similarities.append(description_similarity)
            l4_scores.append(bleu_score)
            
        elif(model_type == 'long'):        
            for j in range(len(pred)):
                bleu_score = calc_bleu_score(tar[j], pred[j])
                if j == 0: l1_scores.append(bleu_score)
                elif j == 1: l2_scores.append(bleu_score)
                elif j == 2: l3_scores.append(bleu_score)
                else: 
                    l4_scores.append(bleu_score)
                    description_similarity = fuzz.token_sort_ratio(tar[j], pred[j])/100
                    description_similarities.append(description_similarity)
                    break
                    
        else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")
            

    return l1_scores, l2_scores, l3_scores, l4_scores, description_similarities


def get_sample_model_name(data_type, model_type):
    if data_type == 'BS':
        if model_type == 'short': model_name = 'v0.3_short_BS_1.320.hdf5'
        else:                     model_name = 'v0.3_long_BS_0.146.hdf5'
    else:
        if model_type == 'short': model_name = 'v0.3_short_IS_0.800.hdf5' 
        else:                     model_name = 'v0.3_long_IS_0.102.hdf5'
            
    return model_name


def load_model(model_path, model_name, device):

    model = torch.load(model_path+model_name, map_location=device)
    model_type = model['model_type']

    max_length = model['max_length']
    input_set = word_encoding.Vocabulary('legacy')
    input_set.__dict__ = model['input_dict']
    output_set = word_encoding.Vocabulary('new')
    output_set.__dict__ = model['output_dict']

    encoder = seq2seq_model.Encoder(input_set.n_words, model['hidden_size']).to(device)

    attention = model['attention']
    if attention == True: 
        decoder = seq2seq_model.AttnDecoder(model['hidden_size'], output_set.n_words, dropout_p=0.1, max_length=max_length).to(device)
    else:
        decoder = seq2seq_model.Decoder(model['hidden_size'], output_set.n_words).to(device)

    encoder.load_state_dict(model['en_sd'])
    decoder.load_state_dict(model['de_sd'])

    encoder_optimizer = model['en_opt']
    decoder_optimizer = model['de_opt']
    encoder_optimizer.load_state_dict(model['en_opt_sd'])
    decoder_optimizer.load_state_dict(model['de_opt_sd'])

    return input_set, output_set, encoder, decoder, max_length, model_type