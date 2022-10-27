# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:37:52 2022

@author: a
"""
from prepro_data import remove_overlap_entities, class_label, event_role_id, event_label_words, ent_label_words, Token_id
import numpy as np
import torch

id2label = dict([(v,k) for k,v in class_label.items()])

def model_compute(in_sent, net, tokenizer, label_mask):
    encode_dict = tokenizer.encode_plus(in_sent, return_tensors='pt')

    batch_arg = encode_dict['input_ids']
    mask_arg = encode_dict['attention_mask']
    token_mask_indices = [np.argwhere(np.array(encode_dict['input_ids']) == 50264)[0][1]]
    label_mask = torch.LongTensor(label_mask)

    batch_arg = batch_arg.cuda()
    mask_arg = mask_arg.cuda()
    label_mask = label_mask.cuda()
    
    preds, out_ans = net(batch_arg, mask_arg, token_mask_indices, Token_id, label_mask)
    
    return preds, out_ans

def model_eval(file_list, net, tokenizer):
    predictions = []
    golds = []
    for inst in file_list:
        tokens = inst['tokens'].copy()
        entities = inst['entity_mentions']
        entities, entity_id_map = remove_overlap_entities(entities)
        entities.sort(key=lambda x: x['start'])
        ent_dict = {} #key为实体的ID，value为（实体的起始位置，实体的结束位置，实体文本，实体类型）
        ent_ids = []
        for entity in entities:
            ent_dict[entity['id']] = (entity['start'], entity['end'], entity['text'], entity['entity_type'])
            ent_ids.append(entity['id'])
        
        ent_ids_inv = ent_ids.copy()
        ent_ids_inv.reverse()
        events = inst['event_mentions']
        events.sort(key=lambda x: x['trigger']['start'])
        for event in events:
            trig_start = event['trigger']['start']
            trig_end = event['trigger']['end']
            trig_text = event['trigger']['text']
            event_type = event['event_type']
            label_mask = [1]*len(class_label)  
            label_mask = np.array(label_mask)
            label_mask[event_role_id[event_type]] = [0]*len(event_role_id[event_type])
            label_mask = label_mask.tolist()
            arguments = event['arguments']
            arg_dict = {} #key为实体的ID，value为元素角色
            for arg in arguments:
                arg_dict[arg['entity_id']] = arg['role']
                      
            sent = inst['sentence']
            pred_for = [] #记录正向的预测结果
            prob_for = [] #记录预测的概率
            for cur_ent_id in ent_ids:
                temp = 'For the '+ event_label_words[event_type]+' event triggered by the '+trig_text #正序的模板
                if cur_ent_id in arg_dict.keys():
                    gold_role = arg_dict[cur_ent_id]

                    if gold_role == 'Place':
                        gold_role = 'Event_'+gold_role
                    else:
                        gold_role = event_type+'_'+gold_role                                       
                else:
                    gold_role = 'Event_None'
                
                if gold_role not in class_label:
                    gold_role = 'Event_None'
                if label_mask[class_label[gold_role]] == 1:
                    gold_role = 'Event_None' 
                    
                golds.append(class_label[gold_role]) 
                    
                for (i,ent_id) in enumerate(ent_ids):
                    (_, _, ent_text, ent_type) = ent_dict[ent_id]
                    new_ent_text = ' or '.join(ent_text)                  
                   
                    if cur_ent_id == ent_id:
                        temp += ', the '+ ent_label_words[ent_type]+', '+new_ent_text+', is <mask>'                      
                        break
                            
                    else:
                        temp += ', the '+ ent_label_words[ent_type]+', '+new_ent_text+', is ' + id2label[pred_for[i]]
                
                in_sent = sent + ' </s> ' + temp #模型的输入
                preds, out_ans = model_compute(in_sent, net, tokenizer, label_mask)
                pred_for.append(preds[0])
                prob_for.append(out_ans[0].unsqueeze(0))
                
            pred_back = [] #记录正向的预测结果
            prob_back = [] #记录预测的概率
            for cur_ent_id in ent_ids_inv:
                temp = 'For the '+ event_label_words[event_type]+' event triggered by the '+trig_text #正序的模板
                    
                for (i,ent_id) in enumerate(ent_ids_inv):
                    (_, _, ent_text, ent_type) = ent_dict[ent_id]
                    new_ent_text = ' or '.join(ent_text)                  
                   
                    if cur_ent_id == ent_id:
                        temp += ', the '+ ent_label_words[ent_type]+', '+new_ent_text+', is <mask>'                      
                        break
                            
                    else:
                        temp += ', the '+ ent_label_words[ent_type]+', '+new_ent_text+', is ' + id2label[pred_back[i]]
                
                in_sent = sent + ' </s> ' + temp #模型的输入
                preds, out_ans = model_compute(in_sent, net, tokenizer, label_mask)
                pred_back.append(preds[0])
                prob_back.append(out_ans[0].unsqueeze(0))
            
            prob_back.reverse()
            prob_for = torch.cat(prob_for, dim=0)
            prob_back = torch.cat(prob_back, dim=0)
            pro_total = prob_for + prob_back
            
            pred_final = torch.argmax(pro_total, dim=1).tolist() # list
            predictions += pred_final
            
            assert len(predictions) == len(golds)

    return predictions, golds