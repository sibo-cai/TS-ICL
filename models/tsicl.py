import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.tsicl_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_TSpfn
import numpy as np


class Model(nn.Module):
    """
    Paper link:
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_x_embedding = DataEmbedding_TSpfn(
            configs.seq_len,
            configs.d_model,
            configs.dropout
        )
        self.enc_y_embedding = DataEmbedding_TSpfn(
            configs.pred_len,
            configs.d_model,
            configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=True
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        # linear
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        # mlp
        # self.projection = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model*2),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model*2, configs.pred_len)
        # )


    def forecast(self, x, y, single_eval_pos):
        # Normalization from Non-stationary Transformer
        # move out of model
        # [batch, n_samples=context_len+1, seq_len]
        # means = x.mean(dim=2, keepdim=True).detach()
        # x = x - means
        # x[torch.abs(x) < 1e-6] = 0
        # stdev = torch.sqrt(
        #     torch.var(x, dim=2, keepdim=True, unbiased=False)
        #     ).detach()
        # stdev = torch.clamp(stdev, min=1e-6)
        # x /= stdev

        # y_train
        # means = y.mean(1, keepdim=True).detach()
        # y = y - means
        # y[torch.abs(y) < 1e-6] = 0
        # stdev = torch.sqrt(torch.var(y, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # y /= stdev

        # y[:, single_eval_pos:, :] = 0.0

        # n_samples = x.shape[1]

        # Embedding
        x_embedding = self.enc_x_embedding(x) # [batch, sample, d_model]
        y_embedding = self.enc_y_embedding(y) # [batch, sample, d_model]

        # x = torch.cat([x_embedding, y_embedding], dim=2)    # [batch, sample, 2 * d_model]
        # following tabicl
        x = x_embedding + y_embedding   # [batch, sample, d_model]

        # split
        x, attns = self.encoder(
            x, single_eval_pos,
        )

        y_test = x[:, single_eval_pos:, :]

        y_test = self.projection(y_test)
        # De-Normalization from Non-stationary Transformer
        # move out of model
        # y_test = y_test * (stdev[:, single_eval_pos:, :].repeat(1, n_samples-single_eval_pos, 1))
        # y_test = y_test + (means[:, single_eval_pos:, :].repeat(1, n_samples-single_eval_pos, 1))
        return y_test, attns

    def forward(self, x_enc, x_dec, single_eval_pos):
        dec_out, attns = self.forecast(x_enc, x_dec, single_eval_pos)
        return dec_out, attns
