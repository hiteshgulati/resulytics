## ATOM Digital Controllership Chart of Accounts
The current process is manual and maps the old account to new client account and then generates the new account description for the old account.
The suggested solution automates the process of generating new account descriptions for the already existing old account descriptions based on the mapping of the source and target account numbers. The developed model can be used to generate recommended account mappings and provides a prediction confidence score, which users can use when considering the recommendations.

### Seq2seq
A Seq2Seq model takes a sequence of items (words, letters, time series, etc) and outputs another sequence of items. The input is a series of words (in this case, old account description), and the output is the translated series of words (new account description).

<p align="center">
<img src="seq2seq.png" alt="drawing" width="500"/></p>

### Data
Data used for training the model includes the data from six companies - Fanatics, JNJ, Energizer CFIN Implementation, Peleton, Republic National Dist, Mars.
Model expects the input client data should be organized in the same format across all the companies. The data should at least contain 3 hierarchical level of information. Data is expected to have the following features.
```
Source Account # | Target Account #	| Source Level Data | Source Account Description | Target Level Data | Target Account Description
```

### Getting Started
To install PyTorch, see installation instructions on the [PyTorch website.](https://pytorch.org/get-started/locally/).

To install all the required dependencies:
```
pip install -r requirements.txt
```
For training the model, tune the hyperparatmeters present in the create_json.py file. Following are the hyperparameters used:
```
'hidden_size': 256,             # number of hidden neurons in NN
'num_iters': 25000,             # number of times a batch of data passed through NN
'benchmark_loss_every': 500,    
'learning_rate': 0.01,          # step size at each iteration while moving toward a min of a loss function
'teacher_forcing_ratio': 0.0,
'max_length': 99                # length of string used as input for encoder
```
Input the excel file name, sheet name, header number in create_json.py file and run it as follows:
```
python create_json.py
```
*(Optional) Once the json file is created, run the training file if you want to train your own model or you can use the trained model present in model directory:
```
python run_training.py config_train.json
```
Run the inference code to predict the result for the input datasheet using the trained model saved in the model folder as follows:
```
python run_inference.py config_infer.json
```
This command will generate a output.xlsx file containing the generated target account description with the confidence score.