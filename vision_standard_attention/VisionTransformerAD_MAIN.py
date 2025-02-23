import random
from torch.nn.modules.fold import F
import tqdm
import numpy as np
import torch
import torch.optim as optim
from TrainValidateWrapper import TrainValidateWrapper
from Models.SimpleTransformer import SimpleTransformer
from PatchEmbedding import PatchEmbedding_CNN
import Utils
import sys
import math
import os
from Logger import Logger
import time
from datetime import datetime
from CSVLogger import CSVLogger

# ------constants------------
#NUM_BATCHES = int(5e5) +10
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 0.5e-4
VALIDATE_EVERY = 1
SEQ_LENGTH = 785 # 14x14 + 1 for cls_token
RESUME_TRAINING = False # set to false to start training from beginning
LAST_BEST_ACCURACY = 0  # Initialize it to zero
#---------------------------

log_file_path = os.path.join("Logs", "training_log.txt")
sys.stdout = Logger(log_file_path)

#---------------------------

#def set_seed(seed):
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available():
        #torch.cuda.manual_seed_all(seed)

#---------------------------

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    This long function is unfortunately doing something very simple and is being 
    very defensive:
    We are separating out all parameters of the model into two buckets: those that 
    will experience
    weight decay for regularization and those that won't (biases, and 
    layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience 
    #regularizing weight decay
    
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding) 
    #,torch.nn.Parameter, torch.nn.Conv2d
    
    for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            #elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
            # # weights of blacklist modules will NOT be weight decayed
            # no_decay.add(fpn)
            elif fpn.startswith('model.token_emb'):
                no_decay.add(fpn)
            # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() -union_params), )
            # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
    return optimizer

def main():
    #set_seed(42)
    global LAST_BEST_ACCURACY
    vision_model = SimpleTransformer(
        dim = 768, # embedding
        num_unique_tokens = 100, # for CIFAR-10, use 100 for CIFAR-100 
        num_layers = 12,
        heads = 8,
        max_seq_len = SEQ_LENGTH,
    ).cuda()
    
    model = TrainValidateWrapper(vision_model)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")
    
     # Initialize CSVLogger with the "Logs" folder
    csv_logger = CSVLogger(folder="Logs", filename="training_metrics.csv")
    
    train_loader, val_loader, testset = Utils.get_loaders_cifar(dataset_type="CIFAR100", img_width=224, img_height=224, batch_size=BATCH_SIZE)
    #optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # optimizer
    optim = configure_optimizers(model)
    
    # Start the timer for throughput calculation
    start_time = time.time()
    
    recent_training_loss = None

    NUM_EPOCHS = 350  # Define number of epochs
    
    # --------training---------
    if RESUME_TRAINING == False:
        start_epoch = 0
    else:
        checkpoint_data = torch.load('checkpoint/visiontrans_model.pt')
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start_epoch = checkpoint_data['epoch']
        
    for epoch in tqdm.tqdm(range(start_epoch, NUM_EPOCHS), desc='Epochs'):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(tqdm.tqdm(train_loader, mininterval=10., desc=f'Training Epoch {epoch+1}')):
            #if i >= 10:  # For faster debugging, limit to 10 batches
                #break
            x = x.cuda()
            y = y.cuda()
            loss = model(x, y)
            loss.backward()
            recent_training_loss = loss.item()  # Store the latest training loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
            
            if i % 1000 == 0:
                print(f'training loss: {loss.item()} -- batch = {i}')
        
        if epoch % VALIDATE_EVERY == 0:
            model.eval()
            total_count = 0
            count_correct = 0
            
            # Start the inference timer
            start_inference_time = time.time()
            
            num_sequences = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.cuda()
                    y = y.cuda()
                    count_correct += model.validate(x, y)
                    total_count += x.shape[0]
                    num_sequences += x.size(0)  # Accumulate the number of sequences (samples) in each batch

                accuracy = (count_correct / total_count) * 100
                print("\n-------------Test Accuracy = ", accuracy, "\n")

                # Stop inference timer
                inference_time = time.time() - start_inference_time

                if accuracy > LAST_BEST_ACCURACY:  # Check if this is the best accuracy
                    LAST_BEST_ACCURACY = accuracy
                    print("----------Best accuracy so far, saving model-----------------")
                    checkpoint_data = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optim.state_dict()
                    }
                    ckpt_path = os.path.join("checkpoint/visiontrans_model_best.pt")
                    torch.save(checkpoint_data, ckpt_path)
            
            # Track GPU memory usage
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            print(f'GPU Memory Allocated: {memory_allocated / 1e6:.2f} MB')
            print(f'GPU Memory Reserved: {memory_reserved / 1e6:.2f} MB')
            
            # Calculate inference time per sample
            inference_time_per_sample = inference_time / num_sequences
            print(f'Inference Time: {inference_time:.6f} seconds')

            # Calculate throughput
            elapsed_time = time.time() - start_time
            
            # Convert elapsed_time to hours, minutes, seconds
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            # Format time as HH:MM:SS for Excel compatibility
            formatted_time = f'{int(hours)}:{int(minutes):02}:{int(seconds):02}'

            print(f'Elapsed Timed: {formatted_time}')
            
            samples_processed = BATCH_SIZE * (i + 1)
            throughput = samples_processed / elapsed_time
            print(f'Throughput: {throughput} samples/second')
            
            # Get current date and time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Log to CSV file
            csv_logger.log(
                current_time=current_time,
                iteration=epoch,
                running_time=formatted_time,
                training_loss=recent_training_loss,
                val_accuracy=accuracy,  # Replace perplexity with accuracy
                learning_rate=optim.param_groups[0]['lr'],
                throughput=throughput,
                gradient_norm=1.0,  # Replace with actual gradient norm calculation if needed
                inference_time=inference_time,
                inference_time_per_sample=inference_time_per_sample,
                memory_allocated=memory_allocated / 1e6,  # Log memory usage in MB
                memory_reserved=memory_reserved / 1e6
            )

        if epoch > 3:  # Learning rate scheduling (example)
            optim.param_groups[0]['lr'] = 0.25e-4

if __name__ == "__main__":
    sys.exit(int(main() or 0))