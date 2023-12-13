# self supervised multimodal multi-task learning network
import os
import sys
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_
from torch.autograd import Variable
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from models.subNets.transformers_encoder.transformer import TransformerEncoder
from models.subNets.BertTextEncoder import BertTextEncoder

__all__ = ['TSCL_FHFN']

class TSCL_FHFN(nn.Module):
    def __init__(self, args):
        super(TSCL_FHFN, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_finetune)

        # audio-vision subnets
        audio_in, video_in = args.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        #self.audio_encoder = nn.Sequential(nn.Linear(args.audio_out, args.audio_out * 2), nn.ReLU(inplace=False))
        #self.video_encoder = nn.Sequential(nn.Linear(args.video_out, args.video_out * 2), nn.ReLU(inplace=False))

        #self.fc_mu_a = nn.Linear(args.audio_out * 2, args.audio_out)
        #self.fc_std_a = nn.Linear(args.audio_out * 2, args.audio_out)

        #self.fc_mu_v = nn.Linear(args.video_out * 2, args.video_out)
        #self.fc_std_v = nn.Linear(args.video_out * 2, args.video_out)

        #self.audio_decoder = nn.Linear(args.audio_out, 1)
        #self.video_decoder = nn.Linear(args.video_out, 1)


        self.project_text_model = nn.Conv1d(args.text_out, args.proj_dim, kernel_size=1, padding=0, bias=False)
        self.project_audio_model = nn.Conv1d(args.audio_out, args.proj_dim, kernel_size=1, padding=0, bias=False)
        self.project_video_model = nn.Conv1d(args.video_out, args.proj_dim, kernel_size=1, padding=0, bias=False)

        self.trans_to_at = TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0)
        self.trans_to_vt = TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0)

        self.trans_to_tv =TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0.1)
        self.trans_to_av =TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0)

        self.trans_to_ta =TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0.1)
        self.trans_to_va =TransformerEncoder(embed_dim=args.proj_dim, num_heads=args.num_heads, layers=args.layers, attn_dropout=0)

        # the post_fusion layers
        # self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        # self.post_fusion_layer_1 = nn.Linear(args.text_out + args.video_out + args.audio_out, args.post_fusion_dim) #self.post_fusion_layer_1 = nn.Linear(args.proj_dim * 6, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

        
        # define the post_fusion layers这是低秩张量融合网络，后面再用
        self.rank = args.rank[0] if args.need_data_aligned else args.rank[1]
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.audio_factor = Parameter(torch.Tensor(self.rank, args.proj_dim * 2 + 1, args.post_fusion_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, args.proj_dim * 2 + 1, args.post_fusion_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, args.proj_dim * 2 + 1, args.post_fusion_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, args.post_fusion_dim))
        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        
        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.post_text_dim)
        self.post_text_layer_2 = nn.Linear(args.post_text_dim, args.post_text_dim)
        self.post_text_layer_3 = nn.Linear(args.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(args.post_audio_dim, args.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(args.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.post_video_dim)
        self.post_video_layer_2 = nn.Linear(args.post_video_dim, args.post_video_dim)
        self.post_video_layer_3 = nn.Linear(args.post_video_dim, 1)

    #def encode_a(self, x):
     #   x1 = self.audio_encoder(x)
     #   return self.fc_mu_a(x1), F.softplus(self.fc_std_a(x1) - 5, beta=1)

   # def encode_v(self, x):
    #    x1 = self.video_encoder(x)
     #   return self.fc_mu_v(x1), F.softplus(self.fc_std_v(x1) - 5, beta=1)

    #def reparameterise(self, mu, std):
    #    eps = torch.randn_like(std)
    #    return mu + std * eps

    def forward(self, text, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach().cpu()
        text = self.text_model(text)[:,0,:]

        if self.aligned:
            audio = self.audio_model(audio, text_lengths)
            video = self.video_model(video, text_lengths)
        else:
            audio = self.audio_model(audio, audio_lengths)
            video = self.video_model(video, video_lengths)

        #mu_a, std_a = self.encode_a(audio)
        #new_audio = self.reparameterise(mu_a, std_a)
        #output_a = self.audio_decoder(new_audio)

        #mu_v, std_v = self.encode_v(video)
        #new_video = self.reparameterise(mu_v, std_v)
        #output_v = self.video_decoder(new_video)

        #project这是统一维度的
        #[batch_size, dim, channels]
        proj_t = self.project_text_model(text.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
        proj_a = self.project_audio_model(audio.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
        proj_v = self.project_video_model(video.unsqueeze(-1)).squeeze(-1).unsqueeze(0)


        feature_at = self.trans_to_at(proj_t, proj_a, proj_a)
        feature_vt = self.trans_to_vt(proj_t, proj_v, proj_v)
        trans_t = torch.cat([feature_at,feature_vt],dim=2)[-1]

        feature_ta = self.trans_to_ta(proj_a, proj_t, proj_t)
        feature_va = self.trans_to_va(proj_a, proj_v, proj_v)
        trans_a = torch.cat([feature_ta,feature_va],dim=2)[-1]

        feature_tv = self.trans_to_tv(proj_v, proj_t, proj_t)
        feature_av = self.trans_to_av(proj_v, proj_a, proj_a)
        trans_v = torch.cat([feature_tv, feature_av], dim=2)[-1]
        
        # fusion这是低秩张量融合网络
        DTYPE = torch.cuda.FloatTensor
        _audio_h = torch.cat((Variable(torch.ones(trans_a.shape[0], 1).type(DTYPE), requires_grad=False), trans_a), dim=1)
        _video_h = torch.cat((Variable(torch.ones(trans_v.shape[0], 1).type(DTYPE), requires_grad=False), trans_v), dim=1)
        _text_h = torch.cat((Variable(torch.ones(trans_t.shape[0], 1).type(DTYPE), requires_grad=False), trans_t), dim=1)
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        output = fusion_audio * fusion_video * fusion_text
        fusion_h = torch.matmul(self.fusion_weights, output.permute(1, 0, 2)).squeeze() + self.fusion_bias
        
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'text': text,
            'audio': audio,
            'video': video,
        }
        return res


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=True):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn1 = nn.LSTM(in_size,hidden_size,num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.rnn2 = nn.LSTM(hidden_size*2,hidden_size,num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size*2, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        packed_h1, _ = self.rnn1(packed_sequence)
        packed_h1_sequence, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = self.layer_norm(packed_h1_sequence)
        packed_norm_sequence = pack_padded_sequence(normed_h1, lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        packed_norm_h1, final_norm_states = self.rnn2(packed_norm_sequence)
        size = final_norm_states[0].size()
        h = self.dropout(final_norm_states[0].view(size[1],-1))
        y_1 = self.linear_1(h)
        
        return y_1
