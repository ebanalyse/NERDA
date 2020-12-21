import numpy as np
from .preprocessing import create_dataloader
from .datasets import get_dane_data
from sklearn import preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import torch
from tqdm import tqdm

def train(model, data_loader, optimizer, device, scheduler, n_tags):
    
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

        # utils.writer.add_scalar('Training Loss', loss,  global_step=i)

    return final_loss / len(data_loader) # Return average loss        

def validate(model, data_loader, device, n_tags):

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
    lfn = torch.nn.CrossEntropyLoss()

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

def enforce_reproducibility(seed = 42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

def train_model(network,
                tag_encoder,
                transformer_tokenizer,
                dataset_training = get_dane_data('train'), 
                dataset_validation = get_dane_data('validate'), 
                max_len = 128,
                train_batch_size = 16,
                validation_batch_size = 8,
                epochs = 5,
                warmup_steps = 0,
                custom_weight_decay = False,
                learning_rate = 5e-5,
                device = None,
                fixed_seed = 42):
    
    if fixed_seed is not None:
        enforce_reproducibility(fixed_seed)
    
    # compute number of unique tags from encoder.
    n_tags = tag_encoder.classes_.shape[0]

    # prepare datasets for modelling by creating data readers and loaders
    # TODO: parametrize num_workers.
    dl_train = create_dataloader(dataset_training.get('sentences'),
                                 dataset_training.get('tags'), 
                                 transformer_tokenizer, 
                                 max_len, 
                                 train_batch_size, 
                                 tag_encoder)
    dl_validate = create_dataloader(dataset_validation.get('sentences'), 
                                    dataset_validation.get('tags'),
                                    transformer_tokenizer, 
                                    max_len, 
                                    validation_batch_size, 
                                    tag_encoder)

    optimizer_parameters = network.parameters()

    # Applying per-parameter weight-decay if chosen
    if custom_weight_decay: 
        param_optimizer = list(network.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    num_train_steps = int(len(dataset_training.get('sentences')) / train_batch_size * epochs)
    
    optimizer = AdamW(optimizer_parameters, lr = learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )

    losses = []
    best_loss = np.inf

    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss = train(network, dl_train, optimizer, device, scheduler, n_tags)
        losses.append(train_loss)
        valid_loss = validate(network, dl_validate, device, n_tags)

        print(f"Train Loss = {train_loss} Valid Loss = {valid_loss}")

        if valid_loss < best_loss:
            best_parameters = network.state_dict()            
            best_loss = valid_loss

    # return best model
    network.load_state_dict(best_parameters)

    return network, losses


        
