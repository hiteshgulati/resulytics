import utils
import torch

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocabulary:

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS, EOS, UNKNOWN

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

            
def create_pairs(x_data, y_data):

    x_data = x_data.reset_index(drop=True)
    y_data = y_data.reset_index(drop=True)

    sources, targets = [], []
    
    for i in range(len(x_data)):
        source_level_text, target_level_text = [], []

        for col in x_data:
            s_text = x_data.loc[i, col]
            source_level_text.append(s_text)
        for col in y_data:
            t_text = y_data.loc[i, col]
            target_level_text.append(t_text)

        sources.append(source_level_text)
        targets.append(target_level_text)

    pairs = list((zip(sources, targets)))
    
    return pairs

def create_vocabulary(set1, set2, x_data, y_data):
    
    input_set = Vocabulary(set1)
    output_set = Vocabulary(set2)
    pairs = create_pairs(x_data, y_data)

    for pair in pairs:
        full_source_string = " ".join(pair[0])
        full_target_string = " ".join(pair[1])
        input_set.add_sentence(full_source_string)
        output_set.add_sentence(full_target_string)

    return input_set, output_set, pairs

def indexes_from_sentence(set, sentence):
    sentence_indexes = []
    for word in sentence.split(' '):
        if word not in set.word2index:
            sentence_indexes.append(2)
        else:
            sentence_indexes.append(set.word2index[word])
            
    return sentence_indexes

def tensor_from_sentence(set, sentence):
    indexes = indexes_from_sentence(set, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_text_list(text_list, input_set, output_set):
    tensors = []
    for level in text_list:
        print(level)
        tensor = tensor_from_sentence(input_set, level)
        tensors.append(tensor)
        
    return tensors

def tensors_from_pair(pair, input_set, output_set):
    source = pair[0]
    target = pair[1]
    source_target_pairwise_tensors = []

    for i in range(len(target)):
        source_level = source[i]
        target_level = target[i]
        source_level_tensor = tensor_from_sentence(input_set, source_level)
        target_level_tensor = tensor_from_sentence(output_set, target_level)
        source_target_pairwise_tensors.append((source_level_tensor, target_level_tensor))

    return source_target_pairwise_tensors