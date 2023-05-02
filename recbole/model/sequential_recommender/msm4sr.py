import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec
import random
import math
from recbole.model.loss import BPRLoss
import os
import numpy as np
from recbole.utils.utils import read_local_pkl
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import VanillaAttention


class ExpertLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(ExpertLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size, eps=1e-12)
        self.apply(self._init_weights)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.norm(self.dropout(self.lin(x)))


class MoELayer(nn.Module):
    """
    MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoELayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([ExpertLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def top_k_gating(self, x):
        clean_logits = x @ self.w_gate
        logits = clean_logits
        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.top_k_gating(x) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class AttnLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.feature_att_layer = VanillaAttention(input_size, hidden_size)

    def forward(self, input_tensor):
        result, _ = self.feature_att_layer(input_tensor)
        return result


class MSM4SR(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda']
        self.weight = config['nip_weight']
        self.mflag = config['mflag']
        self.num_imgtokens = config['num_imgtokens']

        self.proj = config['proj'] # whether to use linear projection for pretrain

        assert self.train_stage in [
            'pretrain', 'finetune'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain']:
            self.item_embedding = None
            # for `finetune`, `item_embedding` is defined in SASRec base model

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
                mapped_feat[i-1, :len(feat), :] = feat

            # # files upload to google drive is generated based on below codes
            # feat_path = os.path.join(config['data_path'], 'desc_emb.pkl')
            # feat_desc = read_local_pkl(feat_path)
            # desc_dim = 768
            # feat_desc_dict = {}
            # for f in feat_desc:
            #     if f[0] in item_list:
            #         if f[0] in feat_desc_dict:
            #             if len(feat_desc_dict[f[0]]) > 9:
            #                 continue
            #             feat_desc_dict[f[0]].append(f[1])
            #         else:
            #             feat_desc_dict[f[0]] = [f[1]]
            # import pickle
            # with open(feat_path, 'wb') as f:
            #     pickle.dump(feat_desc_dict, f)
            # desc_length = max([len(v) for k, v in feat_desc_dict.items()])
            # mapped_feat = np.zeros((self.n_items - 1, desc_length, desc_dim))
            # for i, token in enumerate(dataset.field2id_token['item_id']):
            #     if token == '[PAD]': continue
            #     feat = feat_desc_dict[token]
            #     mapped_feat[i - 1, :len(feat), :] = feat
        else:
            print('Text feature file does not exist!')
            exit(0)

        self.desc_tensor = torch.from_numpy(mapped_feat).float().to(self.device)

        # load image feature
        if os.path.exists(os.path.join(config['data_path'], 'image_emb_'+str(self.num_imgtokens)+'.pkl')):
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

            # # files upload to google drive is generated based on below codes
            # feat_path = os.path.join(config['data_path'], 'image_emb_'+str(self.num_imgtokens)+'.pkl')
            # feat_desc = read_local_pkl(feat_path)
            # img_dim = 768
            # feat_desc_dict = {}
            # for f in feat_desc:
            #     if f[0] in item_list:
            #         if f[0] in feat_desc_dict:
            #             if len(feat_desc_dict[f[0]]) > 9:
            #                 continue
            #             feat_desc_dict[f[0]].append(f[1])
            #         else:
            #             feat_desc_dict[f[0]] = [f[1]]
            # import pickle
            # with open(feat_path, 'wb') as f:
            #     pickle.dump(feat_desc_dict, f)
            # img_length = max([len(v) for k, v in feat_desc_dict.items()])
            # mapped_feat = np.zeros((self.n_items - 1, img_length, img_dim))
            # for i, token in enumerate(dataset.field2id_token['item_id']):
            #     if token == '[PAD]': continue
            #     feat = feat_desc_dict[token]
            #     mapped_feat[i - 1, :len(feat), :] = feat
        else:
            print('Image feature file does not exist!')
            exit(0)

        self.img_tensor = torch.from_numpy(mapped_feat).float().to(self.device)

        # define layers for text encoder
        self.desc_embedding = nn.Embedding(self.n_items, desc_dim, padding_idx=0)
        self.desc_embedding.weight.requires_grad = False
        self.desc_attn = AttnLayer(desc_dim, self.hidden_size)
        self.moe_1 = MoELayer(
            config['n_exps'],
            [desc_dim, self.hidden_size],
            config['adaptor_dropout_prob']
        )

        # define layers for image encoder
        self.img_embedding = nn.Embedding(self.n_items, img_dim, padding_idx=0)
        self.img_embedding.weight.requires_grad = False
        self.img_attn = AttnLayer(img_dim, self.hidden_size)
        self.moe_2 = MoELayer(
            config['n_exps'],
            [img_dim, self.hidden_size],
            config['adaptor_dropout_prob']
        )
        if self.proj:
            self.linear_desc = nn.Linear(self.hidden_size, self.hidden_size)
            self.linear_img = nn.Linear(self.hidden_size, self.hidden_size)

        # Load pre-trained model
        if self.train_stage == 'finetune':
            pretrained_file = config['pretrained_path']
            if pretrained_file is not None:
                checkpoint = torch.load(pretrained_file)
                self.logger.info(f'Loading from {pretrained_file}')
                self.logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                                   k not in ['desc_embedding.weight', 'img_embedding.weight']}
                self.load_state_dict(pretrained_dict, strict=False)
            else:
                pass

    def forward(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'finetune':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def next_item_pred_task(self, seq_output, same_pos_id, pos_item_emb, seq_output_aug=None):
        if seq_output_aug is not None:
            pos_items_emb = F.normalize(pos_item_emb, dim=1)

            pos_logits1 = torch.exp((seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature)
            pos_logits2 = torch.exp((seq_output_aug * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature)

            neg_logits1 = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
            neg_logits2 = torch.matmul(seq_output_aug, pos_items_emb.transpose(0, 1)) / self.temperature

            neg_logits1 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits1)
            neg_logits2 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits2)

            neg_logits1 = torch.exp(neg_logits1).sum(dim=1)
            neg_logits2 = torch.exp(neg_logits2).sum(dim=1)

            loss = -torch.log((pos_logits1 + pos_logits2) / (neg_logits1 + neg_logits2))
        else:
            pos_items_emb = F.normalize(pos_item_emb, dim=1)

            pos_logits1 = torch.exp((seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature)
            # pos_logits2 = torch.exp((seq_output_aug * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature)

            neg_logits1 = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
            # neg_logits2 = torch.matmul(seq_output_aug, pos_items_emb.transpose(0, 1)) / self.temperature

            neg_logits1 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits1)
            # neg_logits2 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits2)

            neg_logits1 = torch.exp(neg_logits1).sum(dim=1)
            # neg_logits2 = torch.exp(neg_logits2).sum(dim=1)

            # loss = -torch.log((pos_logits1 + pos_logits2) / (neg_logits1 + neg_logits2))
            loss = -torch.log((pos_logits1) / (neg_logits1))
        return loss.mean()

    def modality_contrastive_task(self, mod1, mod2, same_pos_id):
        pos_logits = torch.exp((mod1 * mod2).sum(dim=1, keepdim=True) / self.temperature)

        neg_logits1 = torch.matmul(mod1, mod2.transpose(0, 1)) / self.temperature
        neg_logits2 = torch.matmul(mod1, mod1.transpose(0, 1)) / self.temperature
        neg_logits3 = torch.matmul(mod2, mod2.transpose(0, 1)) / self.temperature

        neg_logits1 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits1)
        neg_logits2 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits2)
        neg_logits3 = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits3)

        neg_logits1 = torch.exp(neg_logits1).sum(dim=1)
        neg_logits2 = torch.exp(neg_logits2).sum(dim=1)
        neg_logits3 = torch.exp(neg_logits3).sum(dim=1)

        neg_logits22 = torch.exp((mod1 * mod1).sum(dim=1, keepdim=True) / self.temperature)
        neg_logits33 = torch.exp((mod2 * mod2).sum(dim=1, keepdim=True) / self.temperature)

        loss1 = -torch.log(pos_logits / (neg_logits1 + neg_logits2 - neg_logits22))
        loss2 = -torch.log(pos_logits / (neg_logits1 + neg_logits3 - neg_logits33))
        return (loss1.mean() + loss2.mean())/2.0

    def seq_mixup(self, sequence, emb1, emb2): # desc, img
        if self.mflag:
            return emb1, emb2
        prob = random.uniform(0, 0.5)
        mask = torch.bernoulli(torch.full(emb1.shape[:2], prob))[:, :, None].to(emb1.device)
        masked_items = (sequence == 0)  # mask/pad token should use attr emb
        mask[masked_items] = 0
        reverse_mask = 1 - mask
        result_img = emb1 * mask + emb2 * reverse_mask
        result_desc = emb2 * mask + emb1 * reverse_mask
        return result_desc, result_img

    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_id = interaction['item_id']

        # seq item rand drop
        item_seq, item_seq_len = self.item_seq_rand_drop(item_seq, item_seq_len)

        # aggregate sentence/image feature using an attention layer
        desc_emb_full = self.desc_attn(self.desc_tensor)
        self.desc_embedding.weight.data[1:].copy_(desc_emb_full)
        img_emb_full = self.img_attn(self.img_tensor)
        self.img_embedding.weight.data[1:].copy_(img_emb_full)

        # multimodal feature embeddings of all items in a seq, and adapt by MoE layers
        desc_emb = self.moe_1(self.desc_embedding(item_seq))
        img_emb = self.moe_2(self.img_embedding(item_seq))

        # multimodal seq mixup
        item_seq_mix_desc, item_seq_mix_img = self.seq_mixup(item_seq, desc_emb, img_emb)

        # transformer layers
        seq_output_desc = self.forward(item_seq, item_seq_mix_desc, item_seq_len)
        seq_output_img = self.forward(item_seq, item_seq_mix_img, item_seq_len)

        # multimodal feature projection
        if self.proj:
            seq_output_dd = self.linear_desc(seq_output_desc)
            seq_output_di = self.linear_img(seq_output_desc)
            seq_output_id = self.linear_desc(seq_output_img)
            seq_output_ii = self.linear_img(seq_output_img)

            seq_output_dd = F.normalize(seq_output_dd, dim=1)
            seq_output_di = F.normalize(seq_output_di, dim=1)
            seq_output_id = F.normalize(seq_output_id, dim=1)
            seq_output_ii = F.normalize(seq_output_ii, dim=1)

            # remove sequences with the same next item
            same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0)) # diagonal true others false B, B
            same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device)) # every thing is false

            # positive item embedding after MoE layers
            pos_item_emb_desc = self.moe_1(self.desc_embedding.weight)[pos_id, :]
            pos_item_emb_img = self.moe_2(self.img_embedding.weight)[pos_id, :]

            # next item prediction loss
            loss_seq_item1 = self.next_item_pred_task(seq_output_dd, same_pos_id, pos_item_emb_desc, seq_output_id)
            loss_seq_item2 = self.next_item_pred_task(seq_output_di, same_pos_id, pos_item_emb_img, seq_output_ii)

            # modality contrastive loss
            mloss = self.modality_contrastive_task(seq_output_dd, seq_output_id, same_pos_id)
            mloss1 = self.modality_contrastive_task(seq_output_di, seq_output_ii, same_pos_id)
            return self.weight*loss_seq_item1, self.weight*loss_seq_item2, self.lam * (mloss+mloss1)
        else:
            seq_output_desc = F.normalize(seq_output_desc, dim=1)
            seq_output_img = F.normalize(seq_output_img, dim=1)

            # remove sequences with the same next item
            same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0)) # diagonal true others false B, B
            same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device)) # every thing is false

            # positive item embedding after MoE layers
            pos_item_emb_desc = self.moe_1(self.desc_embedding.weight)[pos_id, :]
            pos_item_emb_img = self.moe_2(self.img_embedding.weight)[pos_id, :]

            # next item prediction loss
            loss_seq_item1 = self.next_item_pred_task(seq_output_desc, same_pos_id, pos_item_emb_desc)
            loss_seq_item2 = self.next_item_pred_task(seq_output_img, same_pos_id, pos_item_emb_img)

            # modality contrastive loss
            mloss = self.modality_contrastive_task(seq_output_desc, seq_output_img, same_pos_id)
            return self.weight*loss_seq_item1, self.weight*loss_seq_item2, self.lam * (mloss)

    def item_seq_rand_drop(self, item_seq, item_seq_len):
        prob = random.uniform(0, 0.2)
        mask_p = torch.full_like(item_seq, 1 - prob, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)
        # Item drop
        seq_mask = item_seq.eq(0).to(torch.bool)
        mask = torch.logical_or(mask, seq_mask)
        mask[:, 0] = True
        drop_index = torch.cumsum(mask, dim=1) - 1

        item_seq_drop = torch.zeros_like(item_seq).scatter(dim=-1, index=drop_index, src=item_seq)
        item_seq_len_aug = torch.gather(drop_index, 1, (item_seq_len - 1).unsqueeze(1)).squeeze() + 1
        return item_seq_drop, item_seq_len_aug

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # left-to-right supervised signals for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # aggregate sentence/image feature using an attention layer
        desc_emb_full = self.desc_attn(self.desc_tensor)
        self.desc_embedding.weight.data[1:].copy_(desc_emb_full)
        img_emb_full = self.img_attn(self.img_tensor)
        self.img_embedding.weight.data[1:].copy_(img_emb_full)

        # text/image features adapted by MoE layers
        desc_emb_full = self.moe_1(self.desc_embedding.weight)
        img_emb_full = self.moe_2(self.img_embedding.weight)

        # multimodal feature embeddings of all items in a seq, and adapt by MoE layers
        desc_emb = self.moe_1(self.desc_embedding(item_seq))
        img_emb = self.moe_2(self.img_embedding(item_seq))

        # transformer layers
        seq_output_desc = self.forward(item_seq, desc_emb, item_seq_len)
        seq_output_img = self.forward(item_seq, img_emb, item_seq_len)
        seq_output_desc = F.normalize(seq_output_desc, dim=-1)
        seq_output_img = F.normalize(seq_output_img, dim=-1)

        # add item ID embeddings during fine-tuning
        if self.train_stage == 'finetune':
            desc_emb_full = desc_emb_full + self.item_embedding.weight
            img_emb_full = img_emb_full + self.item_embedding.weight

        desc_emb_full = F.normalize(desc_emb_full, dim=-1)
        img_emb_full = F.normalize(img_emb_full, dim=-1)

        # compute CE loss
        logits = (torch.matmul(seq_output_desc, desc_emb_full.transpose(0, 1)) + torch.matmul(seq_output_img, img_emb_full.transpose(0, 1))) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # text and image encoders
        desc_emb_full = self.desc_attn(self.desc_tensor)
        self.desc_embedding.weight.data[1:].copy_(desc_emb_full)
        img_emb_full = self.img_attn(self.img_tensor)
        self.img_embedding.weight.data[1:].copy_(img_emb_full)

        desc_emb_full = self.moe_1(self.desc_embedding.weight)
        img_emb_full = self.moe_2(self.img_embedding.weight)

        # multimodal feature embeddings of all items in a seq, and adapt by MoE layers
        desc_emb = self.moe_1(self.desc_embedding(item_seq))
        img_emb = self.moe_2(self.img_embedding(item_seq))

        # transformer layers
        seq_output_desc = self.forward(item_seq, desc_emb, item_seq_len)
        seq_output_img = self.forward(item_seq, img_emb, item_seq_len)

        # add item ID embeddings during fine-tuning
        if self.train_stage == 'finetune':
            desc_emb_full = desc_emb_full + self.item_embedding.weight
            img_emb_full = img_emb_full + self.item_embedding.weight

        # compute the prediction score
        seq_output_desc = F.normalize(seq_output_desc, dim=-1)
        desc_emb_full = F.normalize(desc_emb_full, dim=-1)
        seq_output_img = F.normalize(seq_output_img, dim=-1)
        img_emb_full = F.normalize(img_emb_full, dim=-1)

        scores = torch.matmul(seq_output_desc, desc_emb_full.transpose(0, 1)) + torch.matmul(seq_output_img, img_emb_full.transpose(0, 1))

        # compute CE loss
        logits = scores / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return scores, loss

