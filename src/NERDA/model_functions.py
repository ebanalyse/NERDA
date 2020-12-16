import torch
import torch.nn as nn
from tqdm import tqdm
from .model import n_tags

def train_(model, data_loader, optimizer, device, scheduler):
    
    model.train()    
    final_loss = 0.0
    
    for i, dl in tqdm(enumerate(data_loader), total=len(data_loader)):

        optimizer.zero_grad()
        outputs = model(**dl)
        loss = compute_loss(outputs, 
                            dl.get('target_tags'),
                            dl.get('masks'), 
                            device, 
                            n_tags)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

        # TODO: 
        # Insert gradient acummulation control flow
        # Insert checkpoint-saving flow

        #utils.writer.add_scalar('Training Loss', loss,  global_step=i)

    return final_loss / len(data_loader) # Return average loss        

def validate_(model, data_loader, device):

    model.eval()
    final_loss = 0.0

    for _, dl in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        outputs = model(**dl)
        loss = compute_loss(outputs, 
                            dl.get('target_tags'),
                            dl.get('masks'), 
                            device, 
                            n_tags)
        final_loss += loss.item()

        # TODO: 
        # Insert gradient acummulation control flow
        # Insert checkpoint-saving flow

        #utils.writer.add_scalar('Validation Loss', loss,  global_step=i)
        
    return final_loss / len(data_loader) # Return average loss     

def compute_loss(preds, target_tags, masks, device, n_tags):
    
    # initialize loss function.
    lfn = nn.CrossEntropyLoss()

    # Compute active loss to not compute loss of paddings
    # TODO: elaborate on view function.
    active_loss = masks.view(-1) == 1

    active_logits = preds.view(-1, n_tags)
    active_labels = torch.where(
        active_loss,
        target_tags.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target_tags)
    )

    active_labels = torch.as_tensor(active_labels, device = torch.device(device), dtype = torch.long)
    
    # Only compute loss on actual token predictions
    loss = lfn(active_logits, active_labels)

    return loss