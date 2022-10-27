# coding: UTF-8
#将元素角色对应的语义中每个单词的词向量的均值作为其虚拟label word的初始词向量。
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM
from role_semantic import role_semantic


class RoBERTa_MLM(nn.Module):
    def __init__(self, args, tokenizer):
        super(RoBERTa_MLM, self).__init__()

        self.model = RobertaForMaskedLM.from_pretrained('roberta-base')
        self.model.resize_token_embeddings(len(tokenizer))
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.vocab_size = len(tokenizer)
        
        self.tokenizer = tokenizer    
        self._init_label_word()

    def _init_label_word(self, ):
        
        with torch.no_grad():
            word_embeddings = self.model.get_input_embeddings()
            label_word = list(role_semantic.keys())
            for word in label_word:
                word_id = self.tokenizer.convert_tokens_to_ids(word)
                word_semantic = role_semantic[word]
                word_semantic_id = self.tokenizer(word_semantic, add_special_tokens=False)['input_ids']
                word_embeddings.weight[word_id] = torch.mean(word_embeddings.weight[word_semantic_id], dim=0)
                
            assert torch.equal(self.model.get_input_embeddings().weight, word_embeddings.weight)
      
        
    def forward(self, arg, mask_arg, token_mask_indices, Token_id, label_mask):
        out_arg = self.model(arg, mask_arg)[0].cuda()  # [batch, arg_len, vocab]
        
        out_vocab = torch.zeros(len(arg), self.vocab_size).cuda()
        for i in range(len(arg)):
            out_vocab[i] = out_arg[i][token_mask_indices[i]]  # [arg_len, vocab]
        
        out_ans = out_vocab[:, Token_id] # Tensor.cuda()
        
        label_mask = label_mask.eq(1)
        out_ans.data.masked_fill_(label_mask, -float('inf'))

        # Verbalizer
        pred_word = torch.argmax(out_ans, dim=1).tolist() # list
                
        return pred_word, out_ans
