import argparse
from utils.trainer import train_model
import wandb

def main(args):
    config = {
        'reprocess_input_data': args.reprocess_input_data,
        'overwrite_output_dir': args.overwrite_output_dir,
        'max_seq_length': args.max_seq_length,
        'train_batch_size': args.train_batch_size,
        'num_train_epochs': args.num_train_epochs,
        'save_model_every_epoch': args.save_model_every_epoch,
        'evaluate_generated_text': args.evaluate_generated_text,
        'evaluate_during_training_verbose':args.evaluate_during_training_verbose,
        'use_multiprocessing': args.use_multiprocessing,
        'manual_seed': args.manual_seed,
        'encoder_type1': args.encoder_type1
    }
    train_model(config)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--reprocess_input_data', type=bool, required=False, default=True)
    parser.add_argument('--overwrite_output_dir', type=bool, required=False, default=True)
    parser.add_argument('--max_seq_length', type=int, required=False, default=10)
    parser.add_argument('--train_batch_size', type=int, required=False, default=10)
    parser.add_argument('--num_train_epochs', type=int, required=False, default=10)
    parser.add_argument('--save_model_every_epoch', type=bool, required=False, default=False)
    parser.add_argument('--evaluate_generated_text', type=bool, required=False, default=True)
    parser.add_argument('--evaluate_during_training_verbose', type=bool, required=False, default=True)
    parser.add_argument('--use_multiprocessing', type=bool, required=False, default=False)
    parser.add_argument('--manual_seed', type=bool, required=False, default=4)
    parser.add_argument('--encoder_type1', type=str, required=False, default='roberta')
    arguments = parser.parse_args()
    main(arguments)
