# -*- coding: utf-8 -*-
# @File    : model_wd.py
# @Author  : juntang
# @Time    : 2023/3/4 15:50

# 启用one-hot 方便多类型一起训练
# 增加point-process补充
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from experiment.NYC.exp_args import exp_args


class crime_feature(nn.Module):
    def __init__(self, crime_dim, grid_feature_dim, crime_nhidden, grid_nhidden):
        super(crime_feature, self).__init__()
        self.crime_dim = crime_dim
        self.embedding_crime = nn.Linear(crime_dim, 128, bias=False)
        self.embedding_grid = nn.Linear(grid_feature_dim, 128, bias=False)
        self.hidden_crime = nn.Linear(128, crime_nhidden)
        self.hidden_grid = nn.Linear(128, grid_nhidden)
        self.crime_feature = nn.Sequential(
            nn.Linear(crime_dim, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, crime_nhidden)
        )
        self.grid_feature = nn.Sequential(
            nn.Linear(grid_feature_dim, 128, bias=False),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, grid_nhidden)
        )

    def forward(self, crime_data):
        crime_one_hot = crime_data[:, :, :, :self.crime_dim]
        grid_raw_feature = crime_data[:, :, :, self.crime_dim:]
        crime_out = self.crime_feature(crime_one_hot)
        grid_out = self.grid_feature(grid_raw_feature)
        crime_feature = torch.cat((crime_out, grid_out), -1)

        return crime_feature  # 采用类别和grid的拼接


class crime_similarity(nn.Module):
    def __init__(self, feature_hidden):
        super(crime_similarity, self).__init__()
        self.one_linear = nn.Linear(feature_hidden, feature_hidden, bias=False)
        self.two_linear = nn.Linear(feature_hidden, feature_hidden, bias=False)
        self.similarity_linear = nn.Linear(feature_hidden * 2, feature_hidden)

    def forward(self, feature_1, feature_2):
        one_out = nn.LeakyReLU()(self.one_linear(feature_1))
        two_out = nn.LeakyReLU()(self.one_linear(feature_2))
        out = torch.cat((one_out, two_out), -1)
        out = self.similarity_linear(out)
        # out = nn.LeakyReLU()(self.similarity_linear(out))
        return out


class multi_attention_weight_decay(nn.Module):
    def __init__(self, input_dim, feature_hidden, time_pos_ratio, nhidden=16, embed_dim=16, num_heads=1):
        super(multi_attention_weight_decay, self).__init__()
        assert embed_dim % num_heads == 0
        self.time_pos_ration = time_pos_ratio
        self.embed_dim = embed_dim
        self.embed_dim_k = embed_dim // num_heads
        self.h = num_heads
        self.dim = input_dim  # 类别数量
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim),
                                      nn.Linear(embed_dim, embed_dim)])
        self.time_qk = nn.ModuleList([nn.Linear(embed_dim, embed_dim),
                                      nn.Linear(embed_dim, embed_dim)])
        self.pos_qk = nn.ModuleList([nn.Linear(embed_dim, embed_dim),
                                     nn.Linear(embed_dim, embed_dim)])
        self.val_w = nn.Linear(feature_hidden, feature_hidden)

        self.merge = nn.Linear(input_dim, 1)
        self.new_transfer = nn.Linear(feature_hidden, nhidden)

    def attention_time(self, query, key, type_dim, time_weight_decay, mask=None, dropout=None):
        batch, query_len, d_k = query.size(0), query.size(2), query.size(3)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(type_dim, dim=-1)  # 类别维度
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)  # 对时间进行attention

        time_weight_decay = time_weight_decay.unsqueeze(1)
        p_attn = p_attn * time_weight_decay
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)  # 将0的位置赋值为0，直接softmax被平均了
        if dropout is not None:
            p_attn = dropout(p_attn)
        p_attn = p_attn.unsqueeze(-1)
        return p_attn

    def attention_pos(self, query, key, query_time_len, location_weight_decay, mask=None, dropout=None):
        batch, head, seq_len, type_dim, d_k = key.size()
        query = query.unsqueeze(-2).unsqueeze(-2)
        query = query.repeat(1, 1, seq_len, 1, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.squeeze(-2)
        scores = scores.unsqueeze(2).repeat(1, 1, query_time_len, 1, 1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)  # 对pos进行attention

        location_weight_decay = location_weight_decay.unsqueeze(1)
        p_attn = p_attn * location_weight_decay
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        p_attn = p_attn.unsqueeze(-1)
        return p_attn

    def forward(self, query_t, key_t, query_p, key_p, value, time_weight_decay, location_weight_decay,
                mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, type_dim, f_dim = value.size()
        query_time_len = query_t.size(1)
        if mask is not None:  # h头使用一样的mask
            mask = mask.unsqueeze(1)
        value = self.val_w(value)
        value = value.unsqueeze(1)

        query_t = self.time_qk[0](query_t).view(batch, self.h, -1, self.embed_dim_k)
        key_t = self.time_qk[1](key_t).view(batch, self.h, -1, self.embed_dim_k)
        att_t_scores = self.attention_time(query_t, key_t, type_dim, time_weight_decay, mask, dropout)

        query_p = self.pos_qk[0](query_p).view(batch, self.h, self.embed_dim_k)
        key_p = self.pos_qk[1](key_p).view(batch, self.h, seq_len, type_dim, self.embed_dim_k)
        att_p_scores = self.attention_pos(query_p, key_p, query_time_len, location_weight_decay, mask, dropout)

        scores = self.time_pos_ration * att_t_scores + (1 - self.time_pos_ration) * att_p_scores
        value = value.unsqueeze(-4)
        x = torch.sum(scores * value, -3)
        x = x.transpose(1, 2).contiguous().view(batch, query_time_len, f_dim, type_dim)  # 不进行直接拍平，转化

        x = self.merge(x).squeeze(-1)
        x = self.new_transfer(x)
        return x



class crime_prediction_weight_decay(nn.Module):
    def __init__(self, exp_args, device='cuda'):
        super(crime_prediction_weight_decay, self).__init__()
        freq = exp_args.freq
        num_heads = exp_args.num_heads
        embed_dim = exp_args.embed_dim
        nhidden = exp_args.nhidden
        assert embed_dim % num_heads == 0
        self.exp_args = exp_args
        self.freq = freq
        self.embed_dim = embed_dim
        # self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.feature_hidden = exp_args.grid_feature_hidden + exp_args.crime_one_hot_hidden  # grid 和 one-hot都用
        self.crime_feature_for_time = crime_feature(exp_args.crime_dim, exp_args.grid_feature_dim,
                                                    exp_args.crime_one_hot_hidden, exp_args.grid_feature_hidden)
        self.att = multi_attention_weight_decay(exp_args.time_type_dim, self.feature_hidden, exp_args.time_pos_ratio,
                                                nhidden,
                                                embed_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        self.classifier_concatenate = nn.Sequential(
            nn.Linear(exp_args.query_prediction_time_len * nhidden, nhidden),
            nn.LeakyReLU(),
            nn.Linear(nhidden, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )
        self.risk = nn.Sequential(
            nn.Linear(nhidden, 1)
        )
        self.crime_similarity_for_type = nn.ModuleList([
            crime_similarity(self.feature_hidden) for _ in range(exp_args.crime_dim)
        ])

        self.gru = nn.GRU(nhidden, nhidden)
        self.periodic = nn.Linear(1, embed_dim - 1)
        self.linear = nn.Linear(1, 1)
        self.time_wd_linear = nn.Linear(exp_args.crime_dim, exp_args.crime_dim)
        self.location_wd_linear = nn.Linear(exp_args.crime_dim, exp_args.crime_dim)

    def _get_dis(self, a_g, b_g):
        # time_type_grid [20,1440,24,2]
        # query_grid [20, 2]
        if a_g[0] == -1 or b_g[0] == -1:
            return 1e5 + 5
        return math.sqrt((a_g[0] - b_g[0]) ** 2 + (a_g[1] - b_g[1]) ** 2)

    def _get_dis_wd(self, query_grid, time_type_grid):
        batch, seq_len, type_num, _ = time_type_grid.size()
        qg_x, qg_y = query_grid[:, 0].unsqueeze(-1).unsqueeze(-1), query_grid[:, 1].unsqueeze(-1).unsqueeze(-1)
        qg_x, qg_y = qg_x.repeat(1, seq_len, type_num), qg_y.repeat(1, seq_len, type_num)
        tg_x, tg_y = time_type_grid[:, :, :, 0], time_type_grid[:, :, :, 1]
        dis_x = (qg_x - tg_x) * (qg_x - tg_x)
        dis_y = (qg_y - tg_y) * (qg_y - tg_y)
        dis = torch.sqrt(dis_x + dis_y)
        return dis

    def time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)  # 组合起来embed_dim长

    def pos_embedding(self, pos, is_map=False):
        pos = pos.cpu()
        d_model = self.embed_dim // 2
        if is_map:
            pos_x = pos[:, :, :, 0]
            pos_y = pos[:, :, :, 1]
            pe_x = torch.zeros(pos.shape[0], pos.shape[1], pos.shape[2], d_model)
            pe_y = torch.zeros(pos.shape[0], pos.shape[1], pos.shape[2], d_model)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(self.freq) / d_model))
            position_x = 48. * pos_x.unsqueeze(-1)
            pe_x[:, :, :, 0::2] = torch.sin(position_x * div_term)
            pe_x[:, :, :, 1::2] = torch.cos(position_x * div_term)

            position_y = 48. * pos_y.unsqueeze(-1)
            pe_y[:, :, :, 0::2] = torch.sin(position_y * div_term)
            pe_y[:, :, :, 1::2] = torch.cos(position_y * div_term)
            return torch.cat([pe_x, pe_y], -1)
        else:
            pos_x = pos[:, 0]
            pos_y = pos[:, 1]
            pe_x = torch.zeros(pos.shape[0], d_model)
            pe_y = torch.zeros(pos.shape[0], d_model)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(self.freq) / d_model))

            position_x = 48. * pos_x.unsqueeze(-1)
            pe_x[:, 0::2] = torch.sin(position_x * div_term)
            pe_x[:, 1::2] = torch.cos(position_x * div_term)

            position_y = 48. * pos_y.unsqueeze(-1)
            pe_y[:, 0::2] = torch.sin(position_y * div_term)
            pe_y[:, 1::2] = torch.cos(position_y * div_term)
            return torch.cat([pe_x, pe_y], -1)

    def forward(self, time_type_data, time_type_mask, time_steps, reference_time, time_type_grid, query_grid,
                query_grid_feature, prediction_out_concatenate=False):

        time_type_val = self.crime_feature_for_time(time_type_data)  # batch_size * seq_len * type_num * feature_hidden
        query_grid_val = self.crime_feature_for_time(query_grid_feature.unsqueeze(1).unsqueeze(1))

        # 进行相似表征转换 转为为与查询地点的相似表征向量
        batch, seq_len, type_num, feature_hidden = time_type_val.size()
        sample_len, reference_len = reference_time.size(-2), reference_time.size(-1)
        query_grid_val = query_grid_val.squeeze(1).repeat(1, seq_len, 1)
        for i in range(type_num):
            type_val = time_type_val[:, :, i, :].clone()
            type_similarity = self.crime_similarity_for_type[i](query_grid_val, type_val)
            time_type_val[:, :, i, :] = type_similarity

        tp_mask = time_type_mask[:, -1, -1, :, :].unsqueeze(-1)
        tp_mask = tp_mask.repeat(1, 1, 1, self.feature_hidden)
        time_type_val = time_type_val.masked_fill(tp_mask == 0, 0)

        # location(pos)
        query_pos = self.pos_embedding(query_grid, is_map=False).to(self.device)
        key_pos = self.pos_embedding(time_type_grid, is_map=True).to(self.device)

        # -> location_weight_decay
        location_weight_decay = self._get_dis_wd(query_grid, time_type_grid)
        # location_weight_decay = self.time_wd_linear(location_weight_decay)
        location_weight_decay = location_weight_decay.unsqueeze(1).repeat(1, reference_len, 1, 1)
        location_weight_decay = location_weight_decay.masked_fill(time_type_mask[:, -1, :, :, :] == 0, 10)
        location_weight_decay = torch.softmax(location_weight_decay, dim=-2) * 1000 * -1.0
        location_weight_decay = torch.exp(location_weight_decay).to(self.device)
        location_weight_decay = location_weight_decay.fill_(1)  # todo:去掉权重衰减项
        # time
        key_time = self.time_embedding(time_steps).to(self.device)
        all_out = []
        all_risk_out = torch.zeros((batch, sample_len)) # 现在只去预测时间内的代表点

        for i in range(sample_len):
            n_reference_time = reference_time[:, i, :]
            n_time_type_mask = time_type_mask[:, i, :, :, :]
            query_time = self.time_embedding(n_reference_time).to(self.device)
            # for weight decay
            # -> time_weight_decay
            tp_rt = n_reference_time.clone().unsqueeze(-1)
            tp_ts = time_steps.clone().unsqueeze(-2).repeat(1, reference_len, 1)
            time_weight_decay = torch.abs(tp_ts - tp_rt)
            time_weight_decay = time_weight_decay.unsqueeze(-1).repeat(1, 1, 1, type_num)  # batch * reference_len * seq_len * type_num
            # time_weight_decay = self.time_wd_linear(time_weight_decay)
            time_weight_decay = torch.softmax(time_weight_decay, dim=-2) * 1000 * -1.0
            time_weight_decay = torch.exp(time_weight_decay).to(self.device)
            # 将time_weight_decay 和  location_weight_decay都置为1即为去掉权重衰减项
            time_weight_decay = time_weight_decay.fill_(1)  # todo:去掉权重衰减项
            
            att_out = self.att(query_time, key_time, query_pos, key_pos, time_type_val, time_weight_decay,
                                                  location_weight_decay, n_time_type_mask)
            # att_out: batch_num,time_num, nhidden
            # risk_out = self.risk(att_out).squeeze(-1)
            gru_in = att_out.permute(1, 0, 2)
            _, out = self.gru(gru_in)
            all_out.append(out)
            risk_out = self.risk(out).squeeze(-1).squeeze(0)
            all_risk_out[:, i] = risk_out
            pass

        if prediction_out_concatenate:
            final_out = all_out[0]
            for i in range(1, sample_len):
                final_out = torch.cat([final_out, all_out[i]], -1)
            return self.classifier_concatenate(final_out.squeeze(0)), all_risk_out.to(self.device)
        else:
            for i in range(sample_len - 1):
                all_out[i + 1] = all_out[i + 1] + all_out[i]
            final_out = all_out[sample_len - 1] / sample_len
            # all_risk_out = torch.stack(all_risk_out).transpose(0, 1)  # 转为tensor并交换维度为 batch_num, sample_len, time_num
            return self.classifier(final_out.squeeze(0)), all_risk_out.to(self.device)  # 第i个代表第i个代表点的参考点输出



if __name__ == '__main__':
    pass
    # CUDA_AVAILABLE = True
    # DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")
    # model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)
    #
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
