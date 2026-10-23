import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from VIB import GridVIB


class LayerNorm(nn.Module):

    def __init__(
        self,
        input_dim,
        cond_dim=0,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='xaiver',
        **kwargs
    ):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(
                    in_features=self.cond_dim,
                    out_features=self.hidden_units,
                    bias=False
                )
            if self.center:
                self.beta_dense = nn.Linear(
                    in_features=self.cond_dim,
                    out_features=input_dim,
                    bias=False
                )
            if self.scale:
                self.gamma_dense = nn.Linear(
                    in_features=self.cond_dim,
                    out_features=input_dim,
                    bias=False
                )
        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs


class ConvolutionLayer(nn.Module):

    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU()
        )
        self.convs = nn.ModuleList([nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            dilation=d,
            padding=d
        ) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)
        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f'n_in={self.n_in}, n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'
        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.permute(0, 2, 3, 1)
        return s


class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class ModifiedCoPredictor(nn.Module):

    def __init__(
        self,
        cls_num,
        hid_size,
        biaffine_size,
        channels,
        ffnn_hid_size,
        dropout=0,
        use_entropy_fusion=True
    ):
        super().__init__()
        self.use_entropy_fusion = use_entropy_fusion
        self.biaffine = Biaffine(n_in=hid_size, n_out=biaffine_size, bias_x=True, bias_y=True)
        self.biaffine_linear = nn.Linear(biaffine_size, cls_num)
        self.cnn_channels = channels // 2
        self.vib_channels = channels - self.cnn_channels
        self.mlp_conv = MLP(self.cnn_channels, ffnn_hid_size, dropout=dropout)
        self.linear_conv = nn.Linear(ffnn_hid_size, cls_num)
        self.mlp_vib = MLP(self.vib_channels, ffnn_hid_size, dropout=dropout)
        self.linear_vib = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_confidence(self, logits):
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-08), dim=-1)
        confidence = 1.0 / (1.0 + entropy)
        return confidence.unsqueeze(-1)

    def forward(self, x, y, z):
        grid_states = self.biaffine(x, y)
        o1 = self.biaffine_linear(grid_states)
        cnn_features = z[:, :, :, :self.cnn_channels]
        vib_features = z[:, :, :, self.cnn_channels:]
        z_conv = self.dropout(self.mlp_conv(cnn_features))
        o2 = self.linear_conv(z_conv)
        z_vib = self.dropout(self.mlp_vib(vib_features))
        o3 = self.linear_vib(z_vib)
        if self.use_entropy_fusion:
            conf1 = self.compute_confidence(o1)
            conf2 = self.compute_confidence(o2)
            conf3 = self.compute_confidence(o3)
            total_conf = conf1 + conf2 + conf3 + 1e-08
            w1 = conf1 / total_conf
            w2 = conf2 / total_conf
            w3 = conf3 / total_conf
            weighted_output = o1 * w1 + o2 * w2 + o3 * w3
        else:
            weighted_output = (o1 + o2 + o3) / 3.0
        return (weighted_output, grid_states)


class SAGM(nn.Module):

    def __init__(self, config):
        super(SAGM, self).__init__()
        self.use_bert_last_4_layers = config.use_bert_last_4_layers
        self.lstm_hid_size = config.lstm_hid_size
        self.conv_hid_size = config.conv_hid_size
        self.use_vib = config.use_vib
        self.vib_bottleneck_dim = getattr(config, 'vib_bottleneck_dim', 256)
        self.vib_beta = getattr(config, 'vib_beta', 0.001)
        lstm_input_size = 0
        self.bert = AutoModel.from_pretrained(config.bert_name, output_hidden_states=True)
        lstm_input_size += config.bert_hid_size
        self.dis_embs = nn.Embedding(20, config.dist_emb_size)
        self.reg_embs = nn.Embedding(3, config.type_emb_size)
        self.encoder = nn.LSTM(
            lstm_input_size,
            config.lstm_hid_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        conv_input_size = config.lstm_hid_size + config.dist_emb_size + config.type_emb_size
        self.convLayer = ConvolutionLayer(
            conv_input_size,
            config.conv_hid_size,
            config.dilation,
            config.conv_dropout
        )
        if self.use_vib:
            self.vib_conv = ConvolutionLayer(
                conv_input_size,
                config.conv_hid_size,
                config.dilation,
                config.conv_dropout
            )
            self.vib = GridVIB(
                input_dim=config.conv_hid_size * len(config.dilation),
                bottleneck_dim=config.vib_bottleneck_dim,
                beta=config.vib_beta,
                spatial_pooling='mean'
            )
        self.dropout = nn.Dropout(config.emb_dropout)
        if config.use_vib:
            combined_channels = config.conv_hid_size * len(config.dilation) * 2
        else:
            combined_channels = config.conv_hid_size * len(config.dilation)
        self.predictor = ModifiedCoPredictor(
            config.label_num,
            config.lstm_hid_size,
            config.biaffine_size,
            combined_channels,
            config.ffnn_hid_size,
            config.out_dropout,
            config.use_entropy_fusion
        )
        self.cln = LayerNorm(config.lstm_hid_size, config.lstm_hid_size, conditional=True)

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        bert_outputs = self.bert(
            input_ids=bert_inputs,
            attention_mask=bert_inputs.ne(0).float(),
            output_hidden_states=True
        )
        if self.use_bert_last_4_layers:
            hidden_states = bert_outputs.hidden_states
            bert_embs = torch.stack(hidden_states[-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_outputs.last_hidden_state
        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(
            word_reps,
            sent_length.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(
            packed_outs,
            batch_first=True,
            total_length=sent_length.max()
        )
        cln = self.cln(word_reps.unsqueeze(2), word_reps)
        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)
        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        vib_outputs = None
        if self.use_vib:
            vib_conv_outputs = self.vib_conv(conv_inputs)
            vib_conv_outputs = torch.masked_fill(
                vib_conv_outputs,
                grid_mask2d.eq(0).unsqueeze(-1),
                0.0
            )
            vib_outputs, vib_loss_dict = self.vib(vib_conv_outputs, grid_mask2d)
            combined_outputs = torch.cat([conv_outputs, vib_outputs], dim=-1)
        else:
            combined_outputs = conv_outputs
        outputs, grid_states = self.predictor(word_reps, word_reps, combined_outputs)
        if self.use_vib and vib_outputs is not None:
            vib_outputs = {
                'vib_loss': vib_loss_dict['vib_loss'],
                'vib_kl_loss': vib_loss_dict['kl_loss']
            }
        else:
            vib_outputs = None
        return (outputs, grid_states, vib_outputs)
