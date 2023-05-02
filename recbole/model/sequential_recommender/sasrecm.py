# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss

import os
import numpy as np
from recbole.utils.utils import read_local_pkl


class SASRecM(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecM, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.num_imgtokens = config['num_imgtokens']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # load text feature
        item_list = list(dataset.field2id_token['item_id'])
        if os.path.exists(os.path.join(config['data_path'], 'desc_emb.pkl')):
            feat_path = os.path.join(config['data_path'], 'desc_emb.pkl')
            feat_desc = read_local_pkl(feat_path)
            desc_dim = 768
            feat_desc_dict = {}
            for k, v in feat_desc.items():
                if k in item_list:
                    feat_desc_dict[k] = v
            desc_length = max([len(v) for k, v in feat_desc_dict.items()])
            mapped_feat = np.zeros((self.n_items - 1, desc_length, desc_dim))
            for i, token in enumerate(dataset.field2id_token['item_id']):
                if token == '[PAD]': continue
                feat = feat_desc_dict[token]
                mapped_feat[i - 1, :len(feat), :] = feat
        else:
            print('Text feature file does not exist!')
            exit(0)

        self.desc_tensor = torch.from_numpy(mapped_feat).float().to(self.device)
        self.desc_tensor = torch.mean(self.desc_tensor, 1)

        # load image feature
        if os.path.exists(os.path.join(config['data_path'], 'image_emb_' + str(self.num_imgtokens) + '.pkl')):
            feat_path = os.path.join(config['data_path'], 'image_emb_' + str(self.num_imgtokens) + '.pkl')
            feat_desc = read_local_pkl(feat_path)
            img_dim = 768
            feat_desc_dict = {}
            for k, v in feat_desc.items():
                if k in item_list:
                    feat_desc_dict[k] = v
            img_length = max([len(v) for k, v in feat_desc_dict.items()])
            mapped_feat = np.zeros((self.n_items - 1, img_length, img_dim))
            for i, token in enumerate(dataset.field2id_token['item_id']):
                if token == '[PAD]': continue
                feat = feat_desc_dict[token]
                mapped_feat[i - 1, :len(feat), :] = feat
        else:
            print('Image feature file does not exist!')
            exit(0)

        self.img_tensor = torch.from_numpy(mapped_feat).float().to(self.device)
        self.img_tensor = torch.mean(self.img_tensor, 1)

        self.mlp1 = MLPLayers([768, self.hidden_size, self.hidden_size], 0.2, 'relu', init_method='norm')
        self.mlp2 = MLPLayers([768, self.hidden_size, self.hidden_size], 0.2, 'relu', init_method='norm')

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        item_emb_text = self.mlp1(self.desc_tensor[item_seq])
        item_emb_img = self.mlp2(self.img_tensor[item_seq])

        input_emb = item_emb + position_embedding + item_emb_text + item_emb_img
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[1:] + self.mlp1(self.desc_tensor) + self.mlp2(self.img_tensor)
            test_item_emb = torch.cat((self.item_embedding.weight[0].unsqueeze(0), test_item_emb), dim=0)
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[1:] + self.mlp1(self.desc_tensor) + self.mlp2(self.img_tensor)
        test_items_emb = torch.cat((self.item_embedding.weight[0].unsqueeze(0), test_items_emb), dim=0)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        # compute CE loss
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(scores, pos_items)
        return scores, loss
