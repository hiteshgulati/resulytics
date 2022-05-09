import torch
import random
from tqdm import tqdm
import pandas as pd
import word_encoding
from fuzzywuzzy import fuzz
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import statistics
import utils

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(input_set, output_set, encoder, decoder, account, max_length=100):

    level_confidence_scores = []
    decoded_output = []
    
    with torch.no_grad():
        
        for i in range(len(account)):
            input_level = str(account[i])
            input_level_tensor = word_encoding.tensor_from_sentence(input_set, input_level)
            input_level_length = input_level_tensor.size()[0]
            
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for en in range(input_level_length):
                encoder_output, encoder_hidden = encoder(input_level_tensor[en], encoder_hidden)
                encoder_outputs[en] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden

            decoded_level_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
            #record values for each decoded word
            top_values = []

            for de in range(max_length):
                if(decoder.name == 'AttnDecoder'):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[de] = decoder_attention.data
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
                topv, topi = decoder_output.data.topk(1)
                top_values.append(np.exp(topv.item())*100)

                if topi.item() == EOS_token:
                    decoded_level_words.append('<EOS>')
                    break
                else:
                    #translate decoder output into word and append
                    decoded_level_words.append(output_set.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            decoded_output.append(decoded_level_words)
            level_confidence = statistics.mean(top_values)
            level_confidence_scores.append(level_confidence)
            
        #take average of decoded word values as confidence score
        confidence_score = statistics.mean(level_confidence_scores)

        return decoded_output, decoder_attentions[:de + 1], confidence_score
    
            
def predict_on_unknown(input_set, output_set, encoder, decoder, model_type, df, file_format, max_length=100, verbose=False, progress_bar=True):
    '''
    For a list of legacy accounts, predict a target for each account. Returns a dataframe with the legacy accounts, predicted target accounts, and respective confidence scores.
    '''

    SAN, SL1, SL2, SL3, SAD, _, _, _, _, _ = utils.get_format_names(model_type, have_targets=False, file_format=file_format)
    
    if(model_type == 'short'):
        source_data = {SAN: df[SAN].copy(),
                       SAD: df[SAD].copy()}
        
    elif(model_type == 'long'):     
        source_data = {SAN: df[SAN].copy(), SL1: df[SL1].copy(),
                       SL2: df[SL2].copy(), SL3: df[SL3].copy(), SAD: df[SAD].copy()}
        
    else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")

    sources = pd.DataFrame(source_data)
    
    if(verbose): print('Generating predictions...')
    predictions, confidence_scores = [], []
    
    if(progress_bar):
        for a in tqdm(range(len(sources))):
            account = sources.iloc[a, 1:].tolist()
            #predict on input account
            output, _, confidence_score = predict(input_set, output_set, encoder, decoder, account, max_length=max_length)
            prediction = []
            for pred_level in output:
                pred_level = " ".join(pred_level[:-1]) #drop EOS token
                prediction.append(pred_level)

            predictions.append(prediction)
            confidence_scores.append(confidence_score)
    else:
        for a in range(len(sources)):
            account = sources.iloc[a, 1:].tolist()
            #predict on input account
            output, _, confidence_score = predict(input_set, output_set, encoder, decoder, account, max_length=max_length)
            prediction = []
            for pred_level in output:
                pred_level = " ".join(pred_level[:-1]) #drop EOS token
                prediction.append(pred_level)

            predictions.append(prediction)
            confidence_scores.append(confidence_score)

    df_out = df.copy()
    
    if(model_type == 'short'):
            df_out['Prediction Account Description'] = [p[0] for p in predictions] #remove list brackets from each acct
            df_out['Prediction Account Description'] = df_out['Prediction Account Description'].apply(utils.eliminate_duplicate_words)
            df_out['Confidence Score'] = confidence_scores
    else:     
        df_out['Prediction Level 1'] = list(list(zip(*predictions))[0])
        df_out['Prediction Level 1'] = df_out['Prediction Level 1'].apply(utils.eliminate_duplicate_words)
        
        df_out['Prediction Level 2'] = list(list(zip(*predictions))[1])
        df_out['Prediction Level 2'] = df_out['Prediction Level 2'].apply(utils.eliminate_duplicate_words)
        
        df_out['Prediction Level 3'] = list(list(zip(*predictions))[2])
        df_out['Prediction Level 3'] = df_out['Prediction Level 3'].apply(utils.eliminate_duplicate_words)
        
        df_out['Prediction Account Description'] = list(list(zip(*predictions))[3])
        df_out['Prediction Account Description'] = df_out['Prediction Account Description'].apply(utils.eliminate_duplicate_words)
        
        df_out['Confidence Score'] = confidence_scores
        
    return df_out

def evaluate_output(df_predictions, df_with_targets, model_type, file_format, verbose=True):
    
    _, _, _, _, _, TAN, TL1, TL2, TL3, TAD = utils.get_format_names(model_type, have_targets=True, file_format=file_format)

    if(model_type == 'short'):
        target_data = {TAN: df_with_targets[TAN],
                       TAD: df_with_targets[TAD]}
        
    elif(model_type == 'long'):
        target_data = {TAN: df_with_targets[TAN],
                       TL1: df_with_targets[TL1],
                       TL2: df_with_targets[TL2],
                       TL3: df_with_targets[TL3],
                       TAD: df_with_targets[TAD]}
    
    else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")
    
    df_targets = pd.DataFrame(target_data)

    predictions, targets = [], []
    for i in range(len(df_predictions)):
        if(model_type == 'short'):
            predictions.append(df_predictions.loc[i, 'Prediction Account Description'])
        else:
            predictions.append(df_predictions.loc[i, 'Prediction Level 1':'Prediction Account Description'].tolist())
        targets.append(df_targets.iloc[i, 1:].tolist())
    
    l1_scores, l2_scores, l3_scores, l4_scores, description_similarities = utils.calc_scores(predictions, targets, model_type)
    
    if(model_type == 'short'):
        score_data = {'Predicted Description Bleu Score': l4_scores,
                      'Predicted Description Similarity':description_similarities}
        if(verbose):
             print(f'\nDescription Bleu score average: {np.mean(l4_scores):.2f} \
                     \nAverage predicted description (L4) similarity to target: {np.mean(description_similarities)*100:.2f}%')
    else:
        score_data = {'Predicted Level 1 Bleu Score': l1_scores, 
                      'Predicted Level 2 Bleu Score': l2_scores,
                      'Predicted Level 3 Bleu Score': l3_scores,
                      'Predicted Description Bleu Score': l4_scores,
                      'Predicted Description Similarity':description_similarities}
        if(verbose):
             print(f'\nL1 Bleu score average: {np.mean(l1_scores):.2f} \
                     \nL2 Bleu score average: {np.mean(l2_scores):.2f} \
                     \nL3 Bleu score average: {np.mean(l3_scores):.2f} \
                     \nL4 Bleu score average: {np.mean(l4_scores):.2f} \
                     \nAverage predicted description (L4) similarity to target: {np.mean(description_similarities)*100:.2f}%')
    
    df_scores = pd.DataFrame(score_data)
    
    df_out = pd.concat([df_predictions, df_scores], axis=1)
    
    return df_out
