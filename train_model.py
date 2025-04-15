import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import ray
from ray import tune
import argparse
from Amelia import Amelia
import transformers
from load_data import get_dataloader, load_forbidden_questions_data 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def get_args():
#     parser = argparse.ArgumentParser(description='RuleBERT training loop')
#     parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
#                         help="What optimizer to use")
#     parser.add_argument('--learning_rate', type=float, default=1e-1)
#     parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
#                         help="Whether to use a LR scheduler and what type to use if so")
#     parser.add_argument('--num_warmup_epochs', type=int, default=0,
#                         help="How many epochs to warm up the learning rate for if using a scheduler")
#     parser.add_argument('--max_n_epochs', type=int, default=0,
#                         help="How many epochs to train the model for")
#     parser.add_argument('--patience_epochs', type=int, default=0,
#                         help="If validation performance stops improving, how many epochs should we wait before stopping?")
#     parser.add_argument('--experiment_name', type=str, default='experiment',
#                         help="How should we name this experiment?")
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--test_batch_size', type=int, default=16)

#     args = parser.parse_args()
#     return args

def train_one_epoch(model, optimizer, criterion, dataloader, accumulation_steps):
    model.train()
    running_loss = 0.0

    optimizer.zero_grad() 
    for i, (inputs, attention_mask, targets) in enumerate(tqdm(dataloader, desc='Training')):
        # Move to device
        inputs = inputs.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward
        logits = model(inputs, attention_mask=attention_mask)
        loss = criterion(logits, targets)
        running_loss += loss.item()

        # Scale loss to average over accumulation steps
        loss = loss / accumulation_steps
        loss.backward()  # accumulate gradients

        # Every `accumulation_steps` mini-batches, update weights
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # In case the dataset size isn't divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    # Return average raw loss (not the scaled one)
    return running_loss / len(dataloader)

def evaluate(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(dataloader, desc="Evaluating"):
            # move to device
            input_ids      = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            targets        = targets.to(DEVICE)
            # forward
            outputs = model(input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs, targets)

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def train(config):
    print('Training new Experiment!!')
    print(str(config))
    model = Amelia(num_ffnn_layers=config['num_layers'])
    model.to(DEVICE)

    if config.get('checkpoint_dir', None):
        pass
        #For now there is no checkpoint implementation
    
    num_epochs = config['num_epochs']
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, config['warmup_epochs'], num_epochs)
    train_loader = get_dataloader(config['batch_size'], ray.get(config['train_data']), 'train')
    dev_loader = get_dataloader(config['batch_size'], ray.get(config['dev_data']), 'dev')

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, config['accumulation_steps'])
        val_loss = evaluate(model, criterion, dev_loader)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            exp_name = config.get('exp_name', 'experiment')
            os.makedirs(exp_name, exist_ok=True)
            #hyperparam_items = {k: v for k, v in config.items() if isinstance(v, (int, float, str))}
            #hyperparam_str = "_".join(f"{k}{v}" for k, v in sorted(hyperparam_items.items()))
            file_path = os.path.join(exp_name, f"_best_model.pth")
            torch.save(model.state_dict(), file_path)
            print(f"Saved best model to {file_path} at epoch {epoch+1} with dev loss {val_loss:.4f}")
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        tune.report(train_loss=train_loss, loss=val_loss)


def main():
    train_data, dev_data, test_data = load_forbidden_questions_data()
    ray.init()
    train_dataset_ref = ray.put(train_data)
    dev_dataset_ref = ray.put(dev_data)
    config = {
        'learning_rate': tune.grid_search([5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
        'num_epochs': 20,
        'batch_size': tune.grid_search([1]),
        'num_layers': tune.grid_search([1, 2, 3, 5]),
        'warmup_epochs': tune.grid_search([0, 1, 2]),
        'accumulation_steps': tune.grid_search([4, 8, 16, 32]),
        'train_data': train_dataset_ref,
        'dev_data': dev_dataset_ref
    }

    results = tune.run(
        train,
        config=config,
        resources_per_trial={'cpu': 6, 'gpu': 1}
    )
    print('=== Done with Experiments ===')
    print('Best config: ', results.get_best_config(metric='loss', mode='min'))

if __name__ == '__main__':
    main()