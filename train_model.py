import copy
from statistics import mean, stdev
import time
from math import ceil
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from Amelia import Amelia
import transformers
from load_data import get_dataloader, load_forbidden_questions_data, load_model
from torch.amp import autocast, GradScaler
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = 'chandar-lab/NeoBERT'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = GradScaler() #This probably shouldn't be in global, but its lowkey late
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True) 

def train_one_epoch(model, optimizer, scheduler, dataloader, accumulation_steps):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    optimizer.zero_grad()
    total_batches = len(dataloader)

    
    for i, batch in enumerate(dataloader):
        debug = False
        # Move tensors to device.
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        position_ids = batch['position_ids'].to(DEVICE)
        cu_seqlens = batch['cu_seqlens'].to(DEVICE)
        max_seqlen = batch['max_seqlen']
        targets = batch['labels'].to(DEVICE)

        with autocast(device_type='cuda'):
            outputs = model(input_ids, 
                       attention_mask=attention_mask, 
                       position_ids=position_ids,
                       cu_seqlens=cu_seqlens, 
                       max_seqlen=max_seqlen,
                       labels=targets
                       )
            loss = outputs.loss / accumulation_steps
            logits = outputs.logits
            if debug:
                # print(f'input: ' + str(tokenizer.batch_decode(input_ids.to('cpu'))))
                print(f'Probability: ' + str(torch.softmax(logits, dim=-1)))
                print('Label: ' + str(targets.item()))
        #     print(targets)
        scaler.scale(loss).backward()
        running_loss += loss.item()
        #Stepping through the optizer with ta scaler and clipping gradients
        # have the extra check to make sure that we aren't leaving partial batches behind
        if (i + 1) % accumulation_steps == 0 or i + 1 == total_batches:
            scaler.unscale_(optimizer) # No fucking clue why
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    elapsed_minutes = (time.time() - start_time) / 60 
    print(f'Epoch Training time = {elapsed_minutes:.4f} minutes')
    return (running_loss * accumulation_steps) / len(dataloader)

def evaluate(model, dataloader):
    print('Evaluation Starting')
    eval_start = time.time()
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    pred_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            position_ids = batch['position_ids'].to(DEVICE)
            cu_seqlens = batch['cu_seqlens'].to(DEVICE)
            max_seqlen = batch['max_seqlen']
            targets = batch['labels'].to(DEVICE)

            with autocast(device_type='cuda'):
                outputs = model(input_ids, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens, 
                        max_seqlen=max_seqlen,
                        labels=targets)
                loss = outputs.loss
                logits = outputs.logits 
                # loss = criterion(outputs.logits, targets)

            # convert logits to binary preds (threshold at 0.5)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            y_pred.extend(preds.cpu().tolist())
            y_true.extend(targets.cpu().long().tolist())

    avg_loss = running_loss / len(dataloader)
    f1    = f1_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred)
    rec   = recall_score(y_true, y_pred)
    elapsed_time = (time.time() - eval_start) / 60
    print(f'Evaluation took {elapsed_time} minutes')
    return avg_loss, (prec, rec, f1) 

def train(config):
    print('Training new Experiment!!')
    print(str(config))

    model = load_model()
    # model = model.half()
    model.to(DEVICE)

    num_epochs = config['num_epochs']
    train_loader = get_dataloader(config['batch_size'], config['train_data'], 'train')
    dev_loader = get_dataloader(config['batch_size'], config['dev_data'], 'dev')
    checkpoint_dir = os.path.join('checkpoints', f'{config["experiment_name"]}_experiments')
    os.makedirs(checkpoint_dir, exist_ok=True)

    updates_per_epoch   = ceil(len(train_loader) / config['accumulation_steps'])
    total_updates       = num_epochs * updates_per_epoch
    warmup_updates      = int(config['warmup_epochs'] * updates_per_epoch)

    head_params = [p for n,p in model.named_parameters() if "classifier" in n]
    encoder_params = [p for n,p in model.named_parameters() if "classifier" not in n and p.requires_grad]
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': config['head_lr']},
        {'params': encoder_params, 'lr': config['bert_lr']}
    ])
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'epoch {epoch + 1}')
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, config['accumulation_steps'])
        val_loss, (prec, rec, f1) = evaluate(model, dev_loader)
        print(f'{train_loss=}')
        print(f'{val_loss=}')
        print(
            f"Epoch {epoch}/{num_epochs} "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| P/R/F1: {prec:.3f}/{rec:.3f}/{f1:.3f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"Saved best model to at epoch {epoch+1} with dev loss {val_loss:.4f}")
        else:
            print('Model not saved as it is not the best version of the model')
         
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print(f'Best dev loss was {best_val_loss:.4f}')

def main():
    train_data, dev_data, test_data = load_forbidden_questions_data()
    # ray.init()
    # train_dataset_ref = ray.put(train_data)
    # dev_dataset_ref = ray.put(dev_data)

    # training_config1 = {
    #     'experiment_name': 'XU-3L-1e3lr-16acc-drpt4-epoch5',
    #     'head_lr': 1e-5,
    #     'bert_lr': 5e-6,
    #     'num_epochs': 5,
    #     'batch_size': 1,
    #     'num_layers': 3,
    #     'warmup_epochs': 0.5,
    #     'accumulation_steps': 16,
    #     'dropout': 0.4,
    #     'train_data': train_data,
    #     'dev_data': dev_data,
    # }
    
    # training_config2 = {
    #     'experiment_name': 'XU-3L-2e5lr-16acc-drpt4-epoch5',
    #     'head_lr': 2e-5,
    #     'bert_lr': 5e-6,
    #     'num_epochs': 5,
    #     'batch_size': 1,
    #     'num_layers': 3,
    #     'warmup_epochs': 0.5,
    #     'accumulation_steps': 16,
    #     'dropout': 0.4,
    #     'train_data': train_data,
    #     'dev_data': dev_data, 
    # }
    # # training_configs = [training_config_1, training_config_2]
    
    training_config3 = {
        'experiment_name': '3e5lr-F7',
        'head_lr': 3e-5,
        'bert_lr': 5e-6,
        'num_epochs': 5,
        'batch_size': 1,
        'num_layers': 3,
        'warmup_epochs': 0.5,
        'accumulation_steps': 16,
        'train_data': train_data,
        'dev_data': dev_data, 
    }
    # training_configs = [training_config_1, training_config_2]
    
    # training_config4 = {
    #     'experiment_name': 'XU-3L-5e5lr-16acc-drpt4-epoch5',
    #     'head_lr': 5e-5,
    #     'bert_lr': 5e-6,
    #     'num_epochs': 5,
    #     'batch_size': 1,
    #     'num_layers': 3,
    #     'warmup_epochs': 0.5,
    #     'accumulation_steps': 16,
    #     'dropout': 0.4,
    #     'train_data': train_data,
    #     'dev_data': dev_data,
    # }
    training_configs = [training_config3]
    
    checkpoint_dir = Path("checkpoints").absolute()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # train(training_config)

    for config in training_configs:
        train(config)

    # results = tune.run(
    #     train,
    #     config=training_config,
    #     resources_per_trial={'cpu': 6, 'gpu': 1},
    #     trial_dirname_creator=simple_dirname_creator,
    #     storage_path=storage_path,
    #     name='4-17-Overnight'
    # )

if __name__ == '__main__':
    main()