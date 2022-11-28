# -*- coding: utf-8 -*-

#不加入标识符

import re
from transformers import RobertaTokenizer
from role_semantic import role_semantic, event_role
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

ent_label_words = {'FAC': 'facility', 'ORG': 'organization', 'GPE': 'geographical or political entity', 
             'PER': 'person', 'VEH': 'vehicle', 'WEA': 'weapon', 'LOC': 'location'}

event_label_words = {"Movement:Transport": 'transport', "Personnel:Elect": 'election', "Personnel:Start-Position": 'employment',
               "Personnel:Nominate": 'nomination', "Personnel:End-Position": 'dimission', "Conflict:Attack": 'attack',
               "Contact:Meet": 'meeting', "Life:Marry": 'marriage', "Transaction:Transfer-Money": 'money transfer',
               "Conflict:Demonstrate": 'demonstration', "Business:End-Org": 'collapse', "Justice:Sue": 'prosecution',
               "Life:Injure": 'injury', "Life:Die": 'death', "Justice:Arrest-Jail": 'arrest or jail',
               "Contact:Phone-Write": 'written or telephone communication', 
               "Transaction:Transfer-Ownership": 'ownership transfer', "Business:Start-Org": 'organization founding',
               "Justice:Execute": 'execution', "Justice:Trial-Hearing": 'trial or hearing', "Life:Be-Born": 'birth',
               "Justice:Charge-Indict": 'charge or indict', "Justice:Convict": 'conviction', "Justice:Sentence": 'sentence',
               "Business:Declare-Bankruptcy": 'bankruptcy', "Justice:Release-Parole": 'release or parole', 
               "Justice:Fine": 'fine', "Justice:Pardon": 'pardon', "Justice:Appeal": 'appeal', "Justice:Extradite": 'extradition',
               "Life:Divorce": 'divorce', "Business:Merge-Org": 'organization merger', "Justice:Acquit": 'acquittal'}

ent_label = ent_label_words.keys()
event_label = event_label_words.keys()
add_tokens = [] #添加标识符
for label in ent_label:
    add_tokens.append('<E_START=%s>'%label)
    add_tokens.append('<E_END=%s>'%label)
for label in event_label:
    add_tokens.append('<T_START=%s>'%label)
    add_tokens.append('<T_END=%s>'%label)

tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})

label_word = list(role_semantic.keys())
tokenizer.add_special_tokens({"additional_special_tokens": label_word})

class_label = {} #key为设置的虚拟label word
Token_id = []
i = 0
for word in label_word:
    class_label[word] = i
    i = i+1    
    word_id = tokenizer.convert_tokens_to_ids(word) #label word在预训练语言模型中的id
    Token_id.append(word_id)

event_role_id = {} #key为事件类型， value为角色类型在class_label中的编号,例如[0，1，2]
none_id = class_label['Event_None'] #角色类型为'Event_None'的编号
place_id = class_label['Event_Place'] #角色类型为'Event_Place'的编号
for event in event_label:
    event_role_id[event] = [none_id]
    for role in event_role[event]:
        event_role_id[event].append(class_label[role])

template_new_tokens = ['[V1]', '[V2]', '[V3]', '[V4]', '[V5]', '[V6]']  #temp1
tokenizer.add_special_tokens({"additional_special_tokens": template_new_tokens})

def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity['start'], entity['end']
        for i in range(start, end):
            if tokens[i]:
                id_map[entity['id']] = tokens[i]
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity['id']
    return entities_, id_map

def prepro_data_train (file_list):
    
    input_sents = []
    input_sents_inv = []
    arg_labels = []
    length = []
    arg_num = 0
    label_masks = []
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
            for cur_ent_id in ent_ids:
                temp = event_label_words[event_type] + ' ' + trig_text
                temp_inv = event_label_words[event_type] + ' ' + trig_text

                for ent_id in ent_ids:
                    (_, _, ent_text, ent_type) = ent_dict[ent_id]
                    new_ent_text = ' or '.join(ent_text)
                    if ent_id in arg_dict.keys():
                        ent_role = arg_dict[ent_id]

                        if ent_role == 'Place':
                            ent_role = 'Event_'+ent_role
                        else:
                            ent_role = event_type+'_'+ent_role
                                           
                    else:
                        ent_role = 'Event_None'
                    
                    if ent_role not in class_label:
                        ent_role = 'Event_None'
                    if label_mask[class_label[ent_role]] == 1:
                        ent_role = 'Event_None'                    
                   
                    if cur_ent_id == ent_id:
                        temp += ' ' + ent_label_words[ent_type] + ' ' + new_ent_text + \
                    ' [V1] [V2] [V3] <mask> [V4] [V5] [V6]'
                        arg_labels.append(class_label[ent_role])

                        if ent_role != 'Event_None':
                            arg_num += 1
                        
                        break
                            
                    else:
                        temp +=' ' + ent_label_words[ent_type] + ' ' + new_ent_text + ' ' + ent_role
                
                for ent_id in ent_ids_inv:
                    (_, _, ent_text, ent_type) = ent_dict[ent_id]
                    new_ent_text = ' or '.join(ent_text)
                    if ent_id in arg_dict.keys():
                        ent_role = arg_dict[ent_id]

                        if ent_role == 'Place':
                            ent_role = 'Event_'+ent_role
                        else:
                            ent_role = event_type+'_'+ent_role
                                           
                    else:
                        ent_role = 'Event_None'
                    
                    if ent_role not in class_label:
                        ent_role = 'Event_None'
                    if label_mask[class_label[ent_role]] == 1:
                        ent_role = 'Event_None'                    
                   
                    if cur_ent_id == ent_id:
                        temp_inv += ' ' + ent_label_words[ent_type] + ' ' + new_ent_text + \
                    ' [V1] [V2] [V3] <mask> [V4] [V5] [V6]'
                        
                        break
                            
                    else:
                        temp_inv += ' ' + ent_label_words[ent_type] + ' ' + new_ent_text + ' ' + ent_role
                
                in_sent = sent + ' </s> ' + temp #模型的输入
                input_sents.append(in_sent)            
                
                in_sent_inv = sent + ' </s> ' + temp_inv
                input_sents_inv.append(in_sent_inv)
            
                encode_dict = tokenizer.encode_plus(in_sent)
                input_len = len(encode_dict['input_ids']) #模型的输入长度
                length.append(input_len)
                label_masks.append(label_mask)
    print('Loaded {} instances, including {} argunments'.format(len(arg_labels), arg_num))
    
    return input_sents, input_sents_inv, label_masks, arg_labels, length, tokenizer


