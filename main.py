# -*- coding: utf-8 -*-
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#from sklearn.metrics import f1_score
from load_data import load_data
from scorer import score
from prepro_data import prepro_data_train, Token_id, remove_overlap_entities, tokenizer
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from parameter import parse_args
from model_eval import model_eval

from model import RoBERTa_MLM

torch.cuda.empty_cache()
args = parse_args()  # load parameters


# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


# load data tsv file
train_data = load_data('train.oneie.json')
dev_data = load_data('dev.oneie.json')
test_data = load_data('test.oneie.json')


# get sentence label from data
train_input_sents, train_input_sents_inv, train_label_mask, train_label, train_length, tokenizer = prepro_data_train(train_data)  #Loaded 16649 instances, including 6512 argunments, 最大长度为260
max_len = max(train_length)

label_tr = torch.LongTensor(train_label)

print('Data loaded')


def get_batch(text_data, input_label_mask, indices):
    indices_ids = []
    indices_mask = []
    mask_indices = []  # the place of '<mask>' in 'input_ids'
    label_masks = []

    for idx in indices:
        encode_dict = tokenizer.encode_plus(
            text_data[idx],                 # Prompt 1
            # text_data1[idx] + '</s> ' + ' <mask> ' + text_data2[idx],     # Prompt 2
            # ' <mask> ' + text_data1[idx] + '</s> ' + text_data2[idx],     # Prompt 3
            add_special_tokens=True,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        indices_ids.append(encode_dict['input_ids'])
        indices_mask.append(encode_dict['attention_mask'])
        try: mask_indices.append(np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1])  # id of <mask> is 50264
        except IndexError:
            print(encode_dict['input_ids'])
            print(np.argwhere(np.array(encode_dict['input_ids']) == 50264))
        
        label_mask = torch.LongTensor(input_label_mask[idx])
        label_masks.append(label_mask.unsqueeze(0))

    batch_ids = torch.cat(indices_ids, dim=0)
    batch_mask = torch.cat(indices_mask, dim=0)
    label_masks = torch.cat(label_masks, dim=0)

    return batch_ids, batch_mask, mask_indices, label_masks


# ---------- network ----------
net = RoBERTa_MLM(args, tokenizer).cuda()

# AdamW
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)

criterion = nn.CrossEntropyLoss().cuda()

# creat file to save model and result
file_out = open('./' + args.file_out + '.txt', "w")


print('epoch_num:', args.num_epoch)
print('epoch_num:', args.num_epoch, file=file_out)
print('wd:', args.wd)
print('wd:', args.wd, file=file_out)
print('initial_lr:', args.lr)
print('initial_lr:', args.lr, file=file_out)


##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('Epoch: ', epoch + 1)
    print('Epoch: ', epoch + 1, file=file_out)
    all_indices = torch.randperm(len(train_input_sents)).split(args.batch_size)
    loss_epoch = 0.0
    start = time.time()

    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])
    print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], file=file_out)

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    batch_out = []
    batch_out_inv = []
    batch_truch = []
    for i, batch_indices in enumerate(all_indices, 1):
        # get a batch of wordvecs
        batch_arg, mask_arg, token_mask_indices, label_mask = get_batch(train_input_sents, train_label_mask, batch_indices)
        batch_arg = batch_arg.cuda()
        mask_arg = mask_arg.cuda()
        label_mask = label_mask.cuda()

        truth = Variable(label_tr[batch_indices]).cuda()

        # fed data into network
        _, out_ans = net(batch_arg, mask_arg, token_mask_indices, Token_id, label_mask)
        
        batch_arg, _, _, _ = get_batch(train_input_sents_inv, train_label_mask, batch_indices)
        batch_arg = batch_arg.cuda()

        # fed data into network
        _, out_ans_inv = net(batch_arg, mask_arg, token_mask_indices, Token_id, label_mask)
        
        batch_out.append(out_ans)
        batch_out_inv.append(out_ans_inv)
        batch_truch.append(truth)
        
        if i % args.batch_num == 0 or i == len(all_indices)-1:
            batch_out_tensor_for = torch.cat(batch_out, dim=0)
            batch_out_tensor_inv = torch.cat(batch_out_inv, dim=0)
            batch_truch_tensor = torch.cat(batch_truch, dim=0)

            loss_for = criterion(batch_out_tensor_for, batch_truch_tensor)
            loss_inv = criterion(batch_out_tensor_inv, batch_truch_tensor)
             
          
            loss_all = loss_for + loss_inv   #loss2
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            loss_epoch += loss_all.item()
            if i % (3000 // args.batch_size) == 0 or i == len(all_indices)-1:
                print('loss={:.4f}'.format(loss_epoch / (3000 // args.batch_size)), file=file_out)
                print('loss={:.4f}'.format(loss_epoch / (3000 // args.batch_size)))     
                loss_epoch = 0.0
                
            batch_out = []
            batch_out_inv = []
            batch_truch = []
            del loss_all

    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################

    net.eval()
    with torch.no_grad():
        predictions, golds = model_eval(dev_data, net, tokenizer)        
    dev_p, dev_r, dev_f1 = score(golds,predictions)
        

    # report
    print("epoch {}: dev_p = {:.4f}, dev_r = {:.4f}, dev_f1 = {:.4f}".format(epoch,\
            dev_p, dev_r, dev_f1), file=file_out)
    
    print("epoch {}: dev_p = {:.4f}, dev_r = {:.4f}, dev_f1 = {:.4f}".format(epoch,\
            dev_p, dev_r, dev_f1))
    

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    net.eval()
    with torch.no_grad():
        predictions, golds = model_eval(test_data, net, tokenizer)   
    test_p, test_r, test_f1 = score(golds,predictions)
        

    # report
    print("epoch {}: test_p = {:.4f}, test_r = {:.4f}, test_f1 = {:.4f}".format(epoch,\
            test_p, test_r, test_f1), file=file_out)
    
    print("epoch {}: test_p = {:.4f}, test_r = {:.4f}, test_f1 = {:.4f}".format(epoch,\
            test_p, test_r, test_f1))
    
    saved_model_path = "model" +str(epoch)
    torch.save(net.state_dict(), saved_model_path)

file_out.close()

