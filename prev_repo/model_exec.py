import torch
import torch.nn as nn
from torch.multiprocessing import Process
import argparse

#from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AdamW
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import os
import ast

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

#################################################################
class Distributed_Utils:
    @staticmethod
    # init distributed environment for multiple CPU
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    # handles distribution of data and gradients across the workers
    def model_setup(model, rank):
        #model.to('cpu') #model.to(rank)
        model = DDP(model, find_unused_parameters=True)#, device_ids = [rank])
        return model

    @staticmethod
    def prepare_dataloader(dataset, rank, world_size, batch_size):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return loader

#################################################################
class Utils:

    @staticmethod
    def train(model, train_loader, optimizer, criterion):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training..."):
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, labels) 
            loss.backward()

            torch.distributed.barrier() #ensuring all processes reach here before proceeding.

            optimizer.step()

            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        return average_loss

    @staticmethod
    def compute_metrics(predictions, labels):
        predictions = predictions.detach().cpu().numpy() # if labels and preds are on GPU
        labels = labels.detach().cpu().numpy()

        def calculate_nrmse(y_true, mse, method='range'):
            rmse = np.sqrt(mse)
            if method == 'range':
                normalizer = np.max(y_true) - np.min(y_true)
            elif method == 'mean':
                normalizer = np.mean(y_true)
            elif method == 'std':
                normalizer = np.std(y_true)
            else:
                raise ValueError('invalid method.')

            if normalizer == 0:
                raise ValueError('normalizer cannot be 0.')
            
            nrmse = rmse / normalizer
            return nrmse

        mse = mean_squared_error(labels, predictions)
        nrmse_range = calculate_nrmse(labels, mse, 'range')
        nrmse_mean = calculate_nrmse(labels, mse, 'mean')
        nrmse_std = calculate_nrmse(labels, mse, 'std')

        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        return {
            'MSE': mse,
            'MAE': mae,
            'NRMSE_RANGE': nrmse_range,
            'MRMSE_MEAN': nrmse_mean,
            'NRMSE_STD': nrmse_std,
            'R2': r2
        }

    @staticmethod
    def validate(model, val_loader, criterion):
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation..."):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)

                total_loss += loss.item()
        average_loss = total_loss / len(val_loader)
        return average_loss

    @staticmethod
    def test(model, test_loader, val_loader, criterion):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing..."):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, labels)

                total_loss += loss.item()

                all_predictions.append(predictions)
                all_labels.append(labels)
        
        # concat batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # metrics
        metrics = Utils.compute_metrics(all_predictions, all_labels)
            
        average_loss = total_loss / len(val_loader)
        return average_loss, metrics

    @staticmethod
    def plot_loss_curve(losses_tr, losses_val):
        plt.figure(figsize=(6,4))
        plt.plot(losses_tr, label='Training Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.title('')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig("loss_curve.pdf") 
        plt.show()

    @staticmethod
    def get_splits(csv_fp):
        ''' read csv filenames '''
        patient_file_list = os.listdir(csv_fp)

        ''' split wrt filenames (sample_ids) '''
        # split data into training and remaining data with 80% for training
        training_texts, rem_texts = train_test_split(patient_file_list, test_size=0.2, random_state=42)
        # split the remaining into valid and test sets, 50% each of remaining data
        val_texts, test_texts = train_test_split(rem_texts, test_size=0.5, random_state=42)

        def get_seq_and_label(texts, p_label):
            sequences, labels = [], []

            for i in tqdm(range(len(texts)), desc=p_label):
                df = pd.read_csv(f"{csv_fp}{texts[i]}")
                sequences_in_csv = np.asarray(df['Seq'])
                sequences.append(sequences_in_csv)

                single_label = ast.literal_eval(np.asarray(df['Contributions'])[0])
                labels.append(np.tile(single_label, (len(sequences_in_csv), 1)))

            # size check
            assert(len(sequences) == len(labels))
            assert(len(sequences[0]) == len(labels[0]))

            #flatten
            sequences = np.concatenate(sequences).ravel()
            labels = np.concatenate(labels)
            assert(len(sequences) == len(labels))

            return sequences, labels

        ''' read and merge the dfs that correspond to each split '''
        training_sequences, training_labels = get_seq_and_label(training_texts, 'Training preprocessing...')
        val_sequences, val_labels = get_seq_and_label(val_texts, 'Validation preprocessing...')
        test_sequences, test_labels = get_seq_and_label(test_texts, 'Testing preprocessing...')

        return (training_sequences, training_labels), (val_sequences, val_labels), (test_sequences, test_labels)


#################################################################
class BertForMultiClassClassification(nn.Module):
    def __init__(self, model, num_labels): #num_labels = cell types
        super(BertForMultiClassClassification, self).__init__()
        self.bert = model
        self.regressor = nn.Linear(768, num_labels)
        #self.classifier = nn.Linear(768, num_labels)  #produces logits for each class in a raw way (unnormalized)
        #we will convert into probabilities using softmax

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :] #last_hidden_state
        #logits = self.classifier(cls_output)

        return self.regressor(cls_output)#logits
#################################################################
class DNADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=None):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len or 512 #read length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        return {
            'input_ids' : input_ids,
            'attention_mask': attention_mask,
            'labels':label
        }
#################################################################
def exec_model(rank, world_size, args):
    #print('->', args)
    '''
    path2model = f'DNABERT-2-117M' #'/app/DNABERT-2-117M' #'/home/coguztuzun/confera-analytics/projects/cerag_deconv/git-lfs-3.2.0/DNABERT-2-117M' #'/res-sandbox/compbio/coguztuzun/DNABERT-2-117M' #
    csv_fp = 'cov_data_2/' #'/res-sandbox/compbio/coguztuzun/cov_data_2'
    downsample = True
    
    batch_size = 128
    lr = 5e-5
    num_labels = 12
    epochs = 10
    n = 10000

    patience = 3
    '''
    trigger_times = 0

    if args.verbose: print('model exec initated.')

    try:
        Distributed_Utils.setup(rank, world_size)
        dist.barrier()
        if args.verbose: print(f'rank {rank}: distributed setup completed.')

        #model = Distributed_Utils.model_setup(model, rank)##model.to('cpu')
        
        tokenizer = AutoTokenizer.from_pretrained(args.path_to_model, local_files_only=True, trust_remote_code=True)
        training, val, test = Utils.get_splits(args.path_to_data)

        n_tr = int(args.number_of_reads * 0.7)
        n_val = int((args.number_of_reads - n_tr)/2)

        if n_tr > 0:
            train_dataset = DNADataset(training[0][:n_tr], training[1][:n_tr], tokenizer)
            val_dataset = DNADataset(val[0][:n_val], val[1][:n_val], tokenizer)
            test_dataset = DNADataset(test[0][:n_val], test[1][:n_val], tokenizer)
        else:
            train_dataset = DNADataset(training[0], training[1], tokenizer)
            val_dataset = DNADataset(val[0], val[1], tokenizer)
            test_dataset = DNADataset(test[0], test[1], tokenizer)          

        # dataloader
        train_loader = Distributed_Utils.prepare_dataloader(train_dataset, rank, world_size, args.batch_size)
        val_loader = Distributed_Utils.prepare_dataloader(val_dataset, rank, world_size, args.batch_size)
        test_loader = Distributed_Utils.prepare_dataloader(test_dataset, rank, world_size, args.batch_size)
        dist.barrier()
        if args.verbose: print(f'rank {rank}: data loaders completed.')

        # EXECUTING
        model = AutoModel.from_pretrained(args.path_to_model, local_files_only=True, trust_remote_code=True)
        model = BertForMultiClassClassification(model, num_labels=args.number_of_labels) 
        model = Distributed_Utils.model_setup(model, rank)##model.to('cpu')
        dist.barrier()
        if args.verbose: print(f'rank {rank}: model setup completed.')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate) #AdamW(model.parameters(), lr = lr)
        criterion = nn.MSELoss()
        dist.barrier()
        if args.verbose: print(f'rank {rank}: optimizer + criterion completed.')

        train_losses = []
        val_losses = []

        #early stopping
        best_val_loss = float('inf')

        for epoch in range(args.number_of_epochs):
            train_loss = Utils.train(model, train_loader, optimizer, criterion)
            train_losses.append(train_loss)
            dist.barrier()
            if args.verbose: print(f'rank {rank}: completed training for {epoch+1}.')

            val_loss = Utils.validate(model, val_loader, criterion)
            val_losses.append(val_loss)
            dist.barrier()
            if args.verbose: print(f'rank {rank}: completed val for {epoch+1}.')
            
            if args.verbose: print(f'epoch {epoch + 1}/{args.number_of_epochs}, Train loss: {train_loss}, Val loss: {val_loss}')
            dist.barrier()

            #check-point
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trigger_times = 0
            else:
                trigger_times += 1
            
            if trigger_times >= args.patience:
                if args.verbose: print('early stopping triggered.')
                break
            
        #utils.plot_loss_curve(train_losses, val_losses)

        #TESTESTEST
        test_loss, metrics = Utils.test(model, test_loader, val_loader, criterion)
        if args.verbose: print(f'Test Loss: {test_loss}\nMetrics: {metrics}')
        dist.barrier()
        if args.verbose: print(f'rank {rank}: training loss metrics calculated.')

        if args.verbose: print('model exec ended.')

        # save model 
        if rank == 0:
            torch.save(model.state_dict(), 'BERT_finetuned')
            if args.verbose: print('model saved as BERT_finetuned')

    except Exception as e:
        print(f'Error in process {rank}, {e}')
    finally:
        Distributed_Utils.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Arguments to fine-tune DNABERT data with pre-processed DNA Methylation reads.')
    
    parser.add_argument('-p2m', '--path_to_model', help='Path to local DNABERT repository.', type=str, default='/res-sandbox/compbio/coguztuzun/DNABERT-2-117M')
    parser.add_argument('-p2d', '--path_to_data', help='Path to the directory that has preprocessed BAM files of the patients.', type=str, default='/res-sandbox/compbio/coguztuzun/cov_data_2/')

    parser.add_argument('-ws', '--world_size', help='The number of CPU processes.', type=int, default=3)

    parser.add_argument('-batch_sz', '--batch_size', help='Batch size to use when fine-tuning the model. (Will be removed when hyperparameter tuning is implemented.)', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate to use when fine-tuning the model. (Will be removed when hyperparameter tuning is implemented.)', type=float, default=5e-5)
    parser.add_argument('-nl', '--number_of_labels', help='The number of cell types we have in our labeled data.', type=int, default=12)
    parser.add_argument('-e', '--number_of_epochs', help='The maximum number of epochs to train the model for, early stopping is also implemented.', type=int, default=10)
    parser.add_argument('-p', '--patience', help='Patience parameter for the early stopping.', type=int, default=3)

    parser.add_argument('-n', '--number_of_reads', help='The number of reads, the use of this argument indicates downsampling the training data to the indicated number.', type=int, default=10000)
    parser.add_argument('-rl', '--max_read_len', help='The maximum read length to be used.', type=int, default=512)

    parser.add_argument('-v', '--verbose', help='Verbosity. Prints in-between operations if used.', action='store_false')

    arguments = parser.parse_args()

    if arguments.verbose: print(f'# of processes: {arguments.world_size}')
    mp.set_start_method('fork')
    torch.multiprocessing.spawn(exec_model, args=(arguments.world_size, arguments), nprocs=arguments.world_size, join=True)

if __name__ == "__main__":
    main()

