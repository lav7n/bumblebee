import music21
import time
from music21 import *
from tqdm.notebook import tqdm, trange
import pandas as pd
import logging
import glob
import string
import logging
from utils.Preprocessing import preprocessing
from utils.slidingWindow import sliding_window
from utils.tokeniser import tokenising_data
import logging
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import wandb


def train(reprocess_input_data,overwrite_output_dir,max_seq_length,train_batch_size,num_train_epochs,save_model_every_epoch,evaluate_generated_text,evaluate_during_training_verbose,use_multiprocessing,manual_seed, encoder_type1):
    
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    
    model_args = {
    "reprocess_input_data": reprocess_input_data,
    "overwrite_output_dir": overwrite_output_dir,
    "max_seq_length": max_seq_length,
    "train_batch_size": train_batch_size,
    "num_train_epochs": num_train_epochs,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": save_model_every_epoch,
    "evaluate_generated_text": evaluate_generated_text,
    "evaluate_during_training_verbose": evaluate_during_training_verbose,
    "use_multiprocessing": use_multiprocessing,
    "manual_seed": manual_seed,} 
    encoder_type = encoder_type1
    
    model = Seq2SeqModel(
    encoder_type,
    "roberta-base",
    "bert-base-cased",
    args=model_args,
    use_cuda=True)
    
    run = wandb.init(project="Bumblebee-Transformer", entity="lav7n")
    training_notes,training_duration = preprocessing()
    train,label = sliding_window(training_notes,training_duration)
    training,validation = tokenising_data(train,label)
    results_tr = model.train_model(training, use_wandb = True)
    results_val = model.eval_model(validation, use_wandb = True)
    
       
def train_model(configs):
    train(configs['reprocess_input_data'], configs['overwrite_output_dir'], configs['max_seq_length'],
         configs['train_batch_size'], configs['num_train_epochs'],
         configs['save_model_every_epoch'], configs['evaluate_generated_text'], configs['evaluate_during_training_verbose'],
         configs['use_multiprocessing'], configs['manual_seed'],configs['encoder_type1'])
    
    
    
    
    
