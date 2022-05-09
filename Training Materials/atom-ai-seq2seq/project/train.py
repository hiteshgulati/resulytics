import torch
from torch import optim
import torch.nn as nn
import random
from tqdm import tqdm
import word_encoding
import utils
import os
import numpy as np
import pandas as pd
import inference 
import prepdata

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_iter(tensors_pair, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, model_type, max_length=100, teacher_forcing_ratio=0.0):
    
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    if(model_type == 'long'):
        L1_loss, L2_loss, L3_loss, L4_loss, report_loss = 0, 0, 0, 0, 0
        level_losses = [L1_loss, L2_loss, L3_loss, L4_loss]
        
    elif(model_type=='short'):
        description_loss, report_loss = 0, 0
        level_losses = [description_loss]
        
    else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")
          
    target_lengths = []
    
    for i in range(len(tensors_pair)):
        source_level_tensor = tensors_pair[i][0]
        target_level_tensor = tensors_pair[i][1]
        source_level_length = source_level_tensor.size(0)
        target_level_length = target_level_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        #loop through each word of the source level
        for en in range(source_level_length):
            encoder_output, encoder_hidden = encoder(source_level_tensor[en], encoder_hidden)
            encoder_outputs[en] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for de in range(target_level_length):
                if(decoder.name == 'AttnDecoder'):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                level_losses[i] += criterion(decoder_output, target_level_tensor[de])
                decoder_input = target_level_tensor[de]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for de in range(target_level_length):
                if(decoder.name == 'AttnDecoder'):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                level_losses[i] += criterion(decoder_output, target_level_tensor[de])
                if decoder_input.item() == EOS_token:
                    break
            
        level_losses[i].backward(retain_graph=True)
        target_lengths.append(target_level_length)

    encoder_optimizer.step()
    decoder_optimizer.step()

    #only report loss on description
    report_loss = level_losses[-1].item() / target_lengths[-1]

    return report_loss 


def train_model(input_set, output_set, encoder, decoder, pairs, n_iters, save_path, model_name, model_type, hidden_size, validate, validation_data, abbreviations_file, 
                dropout=None, attention=False, benchmark_every=100, save_every=100, learning_rate=0.01, tf_ratio=0.0, max_length=100, verbose=False):
    
    if (save_every > n_iters) or (benchmark_every > n_iters):
        raise ValueError('Parameter num_iters must be larger than paramters save_every and benchmark_loss_every')
    if n_iters % benchmark_every != 0:
        raise ValueError('Parameter num_iters must be evenly divisible by parameter benchmark_loss_every')

    if(verbose): print('\ntraining model...')
    train_loss_total = 0  # Reset every benchmark_every
    train_losses = []
    val_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    #create training pairs randomly from given list of pairs
    training_pairs = [word_encoding.tensors_from_pair(random.choice(pairs), input_set, output_set) for i in range(n_iters)]
    criterion = nn.NLLLoss()
    
    #prepare validation data
    if(validate):
        df_val = prepdata.format_df(validation_data, model_type, have_targets=True)
        df_val = prepdata.preprocess_df(df_val, abbreviations_file)
    
        if(model_type == 'long'):
            source_val = df_val.loc[:, 'Source Level 1':'Source Account Description']
            target_val = df_val.loc[:, 'Target Level 1':'Target Account Description']
        else:
            source_val = pd.DataFrame({'Source Account Description': df_val['Source Account Description']})
            target_val = pd.DataFrame({'Target Account Description': df_val['Target Account Description']})
             
        val_pairs = word_encoding.create_pairs(source_val, target_val)
        #grab random batch of validation pairs and encode to tensors
        val_tensors_pairs = [word_encoding.tensors_from_pair(random.choice(val_pairs), input_set, output_set) for i in range(250)]
        
    
    #run iteration for each training pair
    for i in tqdm(range(1, n_iters + 1)):
        tensors_pair = training_pairs[i - 1]

        train_loss = run_iter(tensors_pair, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                              criterion, model_type, max_length=max_length, teacher_forcing_ratio=tf_ratio)
        
        train_loss_total += train_loss
        #record loss
        if i % benchmark_every == 0:
            train_loss_avg = train_loss_total / benchmark_every
            train_losses.append(train_loss_avg)
            train_loss_total = 0
            
            if(validate): 
                val_batch_losses = []
                for i in range(len(val_tensors_pairs)):
                    val_tensors_pair = val_tensors_pairs[i]
                    val_loss = validate_pair(val_tensors_pair, input_set, output_set, encoder, decoder, model_type, max_length, criterion)
                    val_batch_losses.append(val_loss)

                val_losses.append(np.mean(val_batch_losses))
             
        #save model parameters
        if i % save_every == 0:
            torch.save({'en_sd': encoder.state_dict(),
                        'de_sd': decoder.state_dict(),
                        'en_opt': encoder_optimizer,
                        'de_opt': decoder_optimizer,
                        'en_opt_sd': encoder_optimizer.state_dict(),
                        'de_opt_sd': decoder_optimizer.state_dict(),
                        'loss': train_losses[-1],
                        'input_dict': input_set.__dict__,
                        'output_dict': output_set.__dict__,
                        'hidden_size': hidden_size,
                        'dropout': dropout, 
                        'max_length': max_length,
                        'attention': attention,
                        'model_type': model_type,
                        }, os.path.join(save_path, '{}_{}_{:.3f}.hdf5'.format(model_name, i, train_losses[-1])), _use_new_zipfile_serialization=False)
            

    utils.plot_loss(model_name, train_losses, val_losses, n_iters, benchmark_every, learning_rate)

    
def validate_pair(val_tensors_pair, input_set, output_set, encoder, decoder, model_type, max_length, criterion):
    
    if(model_type == 'long'):
        L1_loss, L2_loss, L3_loss, L4_loss, report_loss = 0, 0, 0, 0, 0
        level_losses = [L1_loss, L2_loss, L3_loss, L4_loss]
        
    elif(model_type=='short'):
        description_loss, report_loss = 0, 0
        level_losses = [description_loss]
        
    else: raise ValueError("Invalid input for model type: acceptable arguments include ['short', 'long']")
    
    target_lengths = []
    
    with torch.no_grad():       
        
        for i in range(len(val_tensors_pair)):
            source_level_tensor = val_tensors_pair[i][0]
            target_level_tensor = val_tensors_pair[i][1]
            source_level_length = source_level_tensor.size(0)
            target_level_length = target_level_tensor.size(0)

            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for en in range(source_level_length):
                encoder_output, encoder_hidden = encoder(source_level_tensor[en], encoder_hidden)
                encoder_outputs[en] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden

            decoded_level_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for de in range(target_level_length):
                if(decoder.name == 'AttnDecoder'):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    decoder_attentions[de] = decoder_attention.data
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

                level_losses[i] += criterion(decoder_output, target_level_tensor[de])
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze().detach()

                if topi.item() == EOS_token:
                    decoded_level_words.append('<EOS>')
                    break

            target_lengths.append(target_level_length)
                
    report_loss = level_losses[-1].item() / target_lengths[-1]

    return report_loss 
    
    