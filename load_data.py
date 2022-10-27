# -*- coding: utf-8 -*-

import json
import copy

def process(inst):
    new_inst = copy.deepcopy(inst)

    entity_mentions = inst['entity_mentions']
    id_entities = {} #key为实体的ID，value为所有实体提及的文本
    # id_entity = {} #key为实体的ID，value为实体的文本
    for ent_index, entity in enumerate(entity_mentions):
        ent_m_id = entity['id'] #实体提及的id
        id_split = ent_m_id.split('-')
        ent_id = ''
        for k in range(len(id_split)-1):
            if k == 0:
                ent_id = ent_id+id_split[k]
            else:
                ent_id = ent_id +'-'+id_split[k]
        ent = entity['text']
        if ent_id not in id_entities.keys():
            id_entities[ent_id] = ([ent], ent_index, ent_id)
        else:
            id_entities[ent_id][0].append(ent)

                
    new_entity_mentions = []
    for ent_id in id_entities.keys():
        entity_mention = entity_mentions[id_entities[ent_id][1]]
        entity_mention['text'] = id_entities[ent_id][0]
        entity_mention['id'] = id_entities[ent_id][2]
        new_entity_mentions.append(entity_mention)
        
    new_inst['entity_mentions'] = new_entity_mentions
    
    event_mentions = inst['event_mentions']
    for event_index, event in enumerate(event_mentions):
        arguments = event['arguments']
        for arg_index, arg in enumerate(arguments):
            ent_m_id = arg['entity_id']
            id_split = ent_m_id.split('-')
            ent_id = ''
            for k in range(len(id_split)-1):
                if k == 0:
                    ent_id = ent_id+id_split[k]
                else:
                    ent_id = ent_id +'-'+id_split[k]
            arg_text = id_entities[ent_id][0]
            new_inst['event_mentions'][event_index]['arguments'][arg_index]['text'] = arg_text
            new_inst['event_mentions'][event_index]['arguments'][arg_index]['entity_id'] = id_entities[ent_id][2]
        
    return new_inst

def load_data(path, max_length=128, ignore_title=False):
    """Load data from file."""
    overlength_num = title_num = 0
    data = []
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            inst = json.loads(line)
            inst_len = len(inst['pieces'])
            is_title = inst['sent_id'].endswith('-3') \
                and inst['tokens'][-1] != '.' \
                and len(inst['entity_mentions']) == 0
                
            if ignore_title and is_title:
                title_num += 1
                continue
            if max_length != -1 and inst_len > max_length - 2:
                overlength_num += 1
                continue
            if len(inst['entity_mentions']) == 0 or len(inst['event_mentions']) == 0:
                continue
            
            new_inst = process(inst) #以实体而不是实体提及为单位进行元素角色
                                     #对于一个句子中共指的实体提及，我们只考虑对其中文本范围最长的实体提及进行元素角色判断。                      
            data.append(new_inst)
            

    if overlength_num:
        print('Discarded {} overlength instances'.format(overlength_num))
    if title_num:
        print('Discarded {} titles'.format(title_num))
    print('Loaded {} instances from {}'.format(len(data), path))

    return data