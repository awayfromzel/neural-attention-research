import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
from AutoRegressiveWrapper import AutoRegressiveWrapper
from Models.SimpleTransformer import SimpleTransformer
import Utils
import sys
import math
import os
from transformers import AutoTokenizer # pip install transformers
from Logger import Logger
from CSVLogger import CSVLogger
import time
from datetime import datetime


# ------constants------------
NUM_BATCHES = int(2e6) + 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 10000
GENERATE_EVERY = 10000
GENERATE_LENGTH = 512
SEQ_LENGTH = 1024
RESUME_TRAINING = False # set to false to start training from beginning
LAST_BEST_PERPLEXITY = 999
#---------------------------

log_file_path = os.path.join("Logs", "training_log.txt")
sys.stdout = Logger(log_file_path)


#---------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",truncation=True, max_length=1024)

# following commented functions are for character level modeling----------
#def decode_token(token): # convert token to character
# return str(chr(max(32, token)))
#def decode_tokens(tokens): # convert sequence of characters to tokens
# return ''.join(list(map(decode_token, tokens)))
#------------------------------------------------------------------------

def decode_tokens(tokens): # convert token to character - for word level modeling
    return tokenizer.decode(tokens)

def count_parameters(model): # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
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
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decayset!" \
        % (str(param_dict.keys() - union_params), )
        # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
    return optimizer

def compute_perplexity_huggingface(model, test_set, device, max_len=SEQ_LENGTH):
    global LAST_BEST_PERPLEXITY
    stride = 512
    encodings = test_set.data
    encodings = encodings.view(1, encodings.size(0) * encodings.size(1))
    seq_len = encodings.size(1)
    nlls = []
    prev_end_loc = 0
    count = 0
    
    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_len + 1, seq_len + 1)
        if (end_loc - begin_loc) < (max_len + 1):
            continue
        
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        
        if input_ids.shape[-1] < 1025:
            continue
        
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        count = count + 1
        
        # if (count == 50):
        #     break
        
        with torch.no_grad():
            # outputs = model(input_ids, labels=target_ids)  # from hugging face
            loss = model(input_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels to the left by 1.
            # neg_log_likelihood = outputs.loss  # from hugging face

        # nlls.append(neg_log_likelihood)  # from hugging face
        nlls.append(loss)
        
        prev_end_loc = end_loc       
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    best_found = False
    
    if LAST_BEST_PERPLEXITY == 999:
        LAST_BEST_PERPLEXITY = ppl
    else:
        if ppl < LAST_BEST_PERPLEXITY:
            LAST_BEST_PERPLEXITY = ppl
            best_found = True
    
    # save the best model
    print("-----------Perplexity------------- = ", ppl, "---- loss =", torch.stack(nlls).mean())
    
    return best_found

def main():
    simple_model = SimpleTransformer(
        dim = 512, # embedding
        num_unique_tokens = 28996, # for bert-base_cased for wikitext-103,
        # it should be 256 for character level modeling
        num_layers = 8,
        heads = 8,
        max_seq_len = SEQ_LENGTH,
        ).cuda()
    
    model = AutoRegressiveWrapper(simple_model)
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")

    # Initialize CSVLogger with the "Logs" folder
    csv_logger = CSVLogger(folder="Logs", filename="training_metrics.csv")

    train_loader, val_loader, test_loader, val_dataset, test_dataset = Utils.get_loaders_wikitext_103(tokenizer, SEQ_LENGTH, BATCH_SIZE)
    #optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE) # optimizer
    optim = configure_optimizers(model)
    
    # Start the timer for throughput calculation
    start_time = time.time()
    
    recent_training_loss = None
    
    # --------training---------
    if RESUME_TRAINING == False:
        start = 0
    else:
        checkpoint_data = torch.load('Checkpoint/gptamwk_model_best.pt')
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start = checkpoint_data['epoch']
    for i in tqdm.tqdm(range(start, NUM_BATCHES), mininterval = 10., desc = 'training'):
        model.train()
        total_loss = 0
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            loss.backward()
            # Store the latest training loss
            recent_training_loss = loss.item()
            
            # Calculate gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
        if (i % 1000 == 0):
            print(f'training loss: {loss.item()} -- iteration = {i}')
            #print(f'gradient norm: {total_norm}')
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        
        if i % VALIDATE_EVERY == 0:
            model.eval()
            total_len2 = 0
            total_loss2 = 0
            val_count = 1000 # number of validations to compute average BPC
            
            # Track inference time
            start_inference_time = time.time()
            
            with torch.no_grad():
                for v in range(0, val_count):
                    loss = model(next(val_loader))
                    total_loss += loss.item()
                    loss_m = loss.mean()
                    total_loss2 += SEQ_LENGTH * loss_m.item() #loss.float().item() #seq_len
                    total_len2 += SEQ_LENGTH
                #print(f'----------validation loss: {total_loss/val_count}')
                #print(f'Perplexity : {math.exp(total_loss/val_count)}, BPC: {total_loss/val_count*np.log2(2.7173)}')
                #best_found = compute_perplexity_huggingface(model, test_dataset, torch.device('cuda'))
                
                # Store the validation loss before resetting total_loss
                avg_val_loss = total_loss / val_count
                print(f'----------validation loss: {avg_val_loss}')
                
                bpc2 = (total_loss2/total_len2)/math.log(2)
                print("BPC 2 = ", bpc2)
                total_loss = 0
                
            # Stop inference timer
            inference_time = time.time() - start_inference_time
            print(f'Inference Time: {inference_time:.6f} seconds')
            
            # Calculate inference time per sample
			#NOTE: This is incorrect because it does not account for batch size. The total inference time was used to calculate the inference time per sample reported in the paper!
			#The decision was made to leave the code as it was during our experiments and leave this comment in case anyone is curious after trying to recreate our results.
            num_sequences = len(val_dataset)  # Get the number of sequences in the validation set
            inference_time_per_sample = inference_time / num_sequences #This is the line that would need to be updated
            print(f'Inference Time per Sample: {inference_time_per_sample:.6f} seconds')
            
            print(f'Validation Set Size: {len(val_dataset)} sequences')
            
            # Track GPU memory usage
			# NOTE: The memory usage per batch reported in the paper may not exactly match what is printed here.
			# Due to the use of gradient accumulation during training, we adjusted the reported memory usage 
			# after the fact for a fair, apples-to-apples comparison.
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            print(f'GPU Memory Allocated: {memory_allocated / 1e6} MB')
            print(f'GPU Memory Reserved: {memory_reserved / 1e6} MB')

            #Compute perplexity using the new function
            best_found = compute_perplexity_huggingface(model, test_dataset, torch.device('cuda'))

            if best_found:
                print("Best perplexity found! Saving model...")
                checkpoint_data = {
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'optimizer': optim.state_dict()
                }
                ckpt_path = os.path.join("checkpoint/gptamwk_model_best.pt")
                torch.save(checkpoint_data, ckpt_path)
                
            # Log the learning rate
            current_lr = optim.param_groups[0]['lr']
            print(f'Learning rate at iteration {i}: {current_lr}')
            
            # Calculate throughput
            elapsed_time = time.time() - start_time
            
            # Convert elapsed_time to hours, minutes, seconds
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            # Format time as HH:MM:SS for Excel compatibility
            formatted_time = f'{int(hours)}:{int(minutes):02}:{int(seconds):02}'
            print(f'Elapsed Time: {formatted_time}')
            
            samples_processed = BATCH_SIZE * (i + 1)
            throughput = samples_processed / elapsed_time
            print(f'Throughput: {throughput} samples/second')
            
            # Get current date and time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            
            # Log to CSV file
            csv_logger.log(
            current_time=current_time,
            iteration=i,
            running_time=formatted_time,
            training_loss=recent_training_loss,
            val_loss=avg_val_loss,
            perplexity=LAST_BEST_PERPLEXITY.item(),  # NOTE: This should be changed to report the *current* perplexity so that the data can be graphed similar to our paper. We used the log file to get the per-iteration perplexity for our graphs.
            learning_rate=current_lr,
            bpc=bpc2,
            throughput=throughput,
            gradient_norm=total_norm,
            inference_time=inference_time,  # Log total inference time
            inference_time_per_sample=inference_time_per_sample,  # Log inference time per sample
            memory_allocated=memory_allocated / 1e6,  # Log memory usage in MB
            memory_reserved=memory_reserved / 1e6
            )
                
        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            input_start_sequence = decode_tokens(inp)
            print("----------start input------------------")
            print(f'%s \n\n', (input_start_sequence))
            print("----------end of start input-----------")
            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print("----------generated output-------------")
            print(output_str)
            print("----------end generated output---------")
            # ---------save the latest model---------
            #print("----------saving model-----------------")
            #checkpoint_data = {
            #  'epoch': i,
            #  'state_dict': model.state_dict(),
            #  'optimizer': optim.state_dict()
            #}
            #ckpt_path = os.path.join("checkpoint/gptamwk_model.pt")
            #torch.save(checkpoint_data, ckpt_path)
            #revert model to training mode
            model.train()
            
if __name__ == "__main__":
    sys.exit(int(main() or 0))