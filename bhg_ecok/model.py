import torch
from torch import nn
import numpy as np
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    RobertaForMaskedLM,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BartTokenizer,
    BartConfig,
)
from collections import namedtuple
from conv import *


class RobertaClassifier(nn.Module):
    """Fine-tune RoBERTa to directly predict categorical emotions."""

    def __init__(self, args, num_class):
        super(RobertaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(args["model_checkpoint"])
        self.tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"])
        self.config = AutoConfig.from_pretrained(args["model_checkpoint"])
        self.num_future_utts = args["num_future_utterances"]
        hidden_size = self.config.hidden_size

        self.cu_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_class),
        )

    def forward(self, x, mask, utt_pos_mask):
        """
        :param x: The input of PLM. Dim: [B, seq_len, D]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.bert(x, attention_mask=mask)[0]

        utt_xs = x * utt_pos_mask.unsqueeze(-1)
        utt_xs = torch.sum(utt_xs, dim=2) / (
            torch.sum(utt_pos_mask, dim=-1) + 1e-9
        ).unsqueeze(-1)
        print(utt_xs.shape)
        if self.num_future_utts == 0:
            cuu_pos = utt_xs.shape[0] - 1
        else:
            cuu_pos = int((utt_xs.shape[0] - 1) / 2)
        x = utt_xs[cuu_pos, :, :]
        return self.cu_utt_emo_prediction_layers(x)


class CasualRobertaClassifier(nn.Module):
    """Fine-tune RoBERTa to directly predict casual entailment."""

    def __init__(self, args, num_class):
        super(CasualRobertaClassifier, self).__init__()
        self.device = args["device"]
        self.batch_size = args["BATCH_SIZE"]
        self.num_future_utts = args["num_future_utterances"]
        self.comet_hidden_size = args["COMET_HIDDEN_SIZE"]

        # Prepare the encoder and decoder for VAE.
        self.encoder = RobertaModel.from_pretrained(args["model_checkpoint"])
        self.tokenizer = RobertaTokenizer.from_pretrained(args["model_checkpoint"])
        self.config = AutoConfig.from_pretrained(args["model_checkpoint"])

        hidden_size = self.config.hidden_size

        self.cu_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_class),
        )
        self.cause_emo_prediction_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs, mask, utt_pos_mask):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.encoder(inputs.unsqueeze(0), attention_mask=mask.unsqueeze(0))[
            0
        ].squeeze(0)

        utt_xs = x * utt_pos_mask.unsqueeze(-1)
        utt_xs = torch.sum(utt_xs, dim=1) / (torch.sum(utt_pos_mask, dim=-1)).unsqueeze(
            -1
        )

        # print(tgt_features.shape)
        cause_features = torch.cat(
            [utt_xs, utt_xs[-1].unsqueeze(0).repeat(utt_xs.shape[0], 1)], dim=-1
        )

        return (
            self.cu_utt_emo_prediction_layers(utt_xs),
            self.cause_emo_prediction_layers(cause_features).transpose(1, 0).squeeze(0),
        )


class GNN(nn.Module):
    """
        Initialize the HGT model.
    """

    def __init__(
        self,
        in_dim,
        n_hid,
        num_types,
        num_relations,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="hgt",
        prev_norm=True,
        last_norm=True,
        use_RTE=True,
    ):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(
                    conv_name,
                    n_hid,
                    n_hid,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm=prev_norm,
                    use_RTE=use_RTE,
                )
            )
        self.gcs.append(
            GeneralConv(
                conv_name,
                n_hid,
                n_hid,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm=last_norm,
                use_RTE=use_RTE,
            )
        )

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class MultiDimGNN(nn.Module):
    """
    Initialize the MHGT model.
    """

    def __init__(
        self,
        h_dim,
        l_dim,
        num_types,
        num_relations,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="multidim_hgt",
        prev_norm=True,
        last_norm=True,
        use_RTE=True,
    ):
        super(MultiDimGNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.adapt_ws = nn.ModuleList(
            [
                nn.Linear(h_dim, h_dim),
                nn.Linear(l_dim, l_dim),
                nn.Linear(l_dim, l_dim),
                nn.Linear(l_dim, l_dim),
            ]
        )
        self.h_types = [0]
        self.l_types = [1, 2]
        self.drop = nn.Dropout(dropout)
        for l in range(n_layers - 1):
            self.gcs.append(
                MultiDimHGT(
                    h_dim,
                    l_dim,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm=prev_norm,
                    use_RTE=use_RTE,
                )
            )
        self.gcs.append(
            MultiDimHGT(
                h_dim,
                l_dim,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm=last_norm,
                use_RTE=use_RTE,
            )
        )

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.h_dim).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if t_id in self.h_types:
                d = self.h_dim
            else:
                d = self.l_dim
            if idx.sum() == 0:
                continue
            res[idx, :d] = torch.tanh(self.adapt_ws[t_id](node_feature[idx, :d]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


class VIBERC(nn.Module):
    """The BHG model for ERC task."""

    def __init__(self, args, num_class):
        super(VIBERC, self).__init__()
        self.device = args["device"]
        self.batch_size = args["BATCH_SIZE"]
        self.num_future_utts = args["num_future_utterances"]
        self.comet_hidden_size = args["COMET_HIDDEN_SIZE"]
        self.win_size = (
            args["COMET_WIN_SIZE"] + 1
            if self.num_future_utts == 0
            else 2 * args["COMET_WIN_SIZE"] + 1
        )
        self.conceptnet = "conceptnet" in args["ROOT_DIR"]

        # Prepare the encoder and decoder for VAE.
        self.encoder = RobertaModel.from_pretrained(args["model_checkpoint"])
        self.tokenizer = RobertaTokenizer.from_pretrained(args["model_checkpoint"])
        self.config = AutoConfig.from_pretrained(args["model_checkpoint"])

        hidden_size = self.config.hidden_size
        self.conv_name = args["CONV_NAME"]

        if self.conv_name == "multidim_hgt":
            self.gnn = MultiDimGNN(
                h_dim=hidden_size,
                l_dim=self.comet_hidden_size,
                n_heads=8,
                n_layers=2,
                dropout=self.config.hidden_dropout_prob,
                num_types=4,
                num_relations=4,
                use_RTE=False,
            )
        else:
            self.gnn = GNN(
                conv_name=self.conv_name,
                in_dim=hidden_size,
                n_hid=hidden_size,
                n_heads=8,
                n_layers=2,
                dropout=self.config.hidden_dropout_prob,
                num_types=4,
                num_relations=4,
                use_RTE=False,
            )

        self.context2params = nn.ModuleDict()

        """The variational modules for target utterance representations."""
        utt_layer = nn.Linear(hidden_size + 2 * self.comet_hidden_size, 2 * hidden_size)
        self.context2params["utt"] = utt_layer

        # self.comet_utt_layer = nn.Linear(
        #    hidden_size, comet_hidden_size)

        """The variational modules for comet utterance-level knowledge representations."""
        """self.comet_attn_layer = nn.Linear(hidden_size, comet_hidden_size)
        comet_utt_layer = nn.Linear(
            comet_hidden_size, 2 * comet_hidden_size)
        self.context2params["comet_utt"] = comet_utt_layer"""

        """self.comet_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(comet_hidden_size, comet_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(comet_hidden_size, num_class))"""

        self.emb = RelTemporalEncoding(hidden_size)
        # self.know_emb = RelTemporalEncoding(comet_hidden_size)
        self.forward_pos = torch.LongTensor([i for i in range(5)]).to(self.device)
        self.backward_pos = torch.LongTensor([i for i in range(5, 7)]).to(self.device)
        # self.comet_utt_f = nn.Parameter(torch.zeros(1, self.win_size, self.comet_hidden_size))
        # self.comet_utt_b = nn.Parameter(torch.zeros(1, self.win_size, self.comet_hidden_size))

        self.cu_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_class),
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        :param Q: [B, WIN, D]
        :param K: [B, WIN, 9, D]
        :param V: [B, WIN, 9, D]
        :param mask: [B, WIN]
        :return: Attended results.
        """
        attn_weights = torch.matmul(K, Q.unsqueeze(-1)).squeeze(-1) / torch.sqrt(
            torch.tensor(Q.shape[-1])
        )
        if mask is not None:
            mask = (1.0 - mask) * -100000.0
            attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights.unsqueeze(-2), V).squeeze(-2)

    def compute_latent_params(self, context, layer, mode="train"):
        """Estimate the latent parameters."""
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        params = layer(context)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.tanh(logvar)
        if mode == "train":
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            z = mu

        return Params(z, mu, logvar)

    def keepdim_make_graph(
        self,
        utt_feature,
        utt_mask,
        agg_feature_forward,
        agg_feature_back,
        comet_feature,
    ):
        """
                :param utt_feature: [B, WIN, D1]
                :param utt_mask: [B, WIN]
                :param agg_feature: [B, WIN, D2]
                :param comet_feature: [B, WIN, 9, D2]
                :return:
        """
        if self.conceptnet:
            connect_matrix = torch.sum(comet_feature, dim=-1) != 0
        agg_feature_forward = torch.cat(
            [
                agg_feature_forward,
                torch.zeros(
                    [
                        agg_feature_forward.shape[0],
                        agg_feature_forward.shape[1],
                        utt_feature.shape[-1] - agg_feature_forward.shape[2],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        agg_feature_back = torch.cat(
            [
                agg_feature_back,
                torch.zeros(
                    [
                        agg_feature_back.shape[0],
                        agg_feature_back.shape[1],
                        utt_feature.shape[-1] - agg_feature_back.shape[2],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        comet_feature = torch.cat(
            [
                comet_feature,
                torch.zeros(
                    [
                        comet_feature.shape[0],
                        comet_feature.shape[1],
                        comet_feature.shape[2],
                        utt_feature.shape[-1] - comet_feature.shape[3],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature_forward,
                agg_feature_back,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * utt_feature.shape[1]
                + [3] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            for j in range(utt_feature.shape[1]):
                if utt_mask[i, j] == 0:
                    continue
                cuu_source.append(j)
                cuu_target.append(j + utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

                cuu_source.append(j)
                cuu_target.append(j + 2 * utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                if utt_mask[i, j - utt_feature.shape[1]] == 0:
                    continue
                for k in range(j - utt_feature.shape[1], utt_feature.shape[1]):
                    if utt_mask[i, k] == 0:
                        continue
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(2)

            for j in range(2 * utt_feature.shape[1], 3 * utt_feature.shape[1]):
                if utt_mask[i, j - 2 * utt_feature.shape[1]] == 0:
                    continue
                for k in range(0, j - 2 * utt_feature.shape[1] + 1):
                    if utt_mask[i, k] == 0:
                        continue
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(2)
                    edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                if utt_mask[i, j - utt_feature.shape[1]] == 0:
                    continue
                s = 3 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    if self.conceptnet:
                        if connect_matrix[i, j - utt_feature.shape[1], k - s]:
                            cuu_source.append(k)
                            cuu_target.append(j)
                            edge_type.append(3)
                            edge_time.append(0)

                            cuu_source.append(k)
                            cuu_target.append(j + utt_feature.shape[1])
                            edge_type.append(3)
                            edge_time.append(0)
                    else:
                        cuu_source.append(k)
                        cuu_target.append(j)
                        edge_type.append(3)
                        edge_time.append(0)

                        cuu_source.append(k)
                        cuu_target.append(j + utt_feature.shape[1])
                        edge_type.append(3)
                        edge_time.append(0)

            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def make_graph(
        self,
        utt_feature,
        utt_mask,
        agg_feature_forward,
        agg_feature_back,
        comet_feature,
    ):
        """
        :param utt_feature: [B, WIN, D]
        :param utt_mask: [B, WIN]
        :param agg_feature_forward: [B, WIN, D]
        :param agg_feature_back: [B, WIN, D]
        :param comet_feature: [B, WIN, 9, D]
        :return:
        """
        if self.conceptnet:
            connect_matrix = torch.sum(comet_feature, dim=-1) != 0
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature_forward,
                agg_feature_back,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * utt_feature.shape[1]
                + [3] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            for j in range(utt_feature.shape[1]):
                if utt_mask[i, j] == 0:
                    continue
                cuu_source.append(j)
                cuu_target.append(j + utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

                cuu_source.append(j)
                cuu_target.append(j + 2 * utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                if utt_mask[i, j - utt_feature.shape[1]] == 0:
                    continue
                for k in range(j - utt_feature.shape[1], utt_feature.shape[1]):
                    if utt_mask[i, k] == 0:
                        continue
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(2)

            for j in range(2 * utt_feature.shape[1], 3 * utt_feature.shape[1]):
                if utt_mask[i, j - 2 * utt_feature.shape[1]] == 0:
                    continue
                for k in range(0, j - 2 * utt_feature.shape[1] + 1):
                    if utt_mask[i, k] == 0:
                        continue
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(2)
                    edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                if utt_mask[i, j - utt_feature.shape[1]] == 0:
                    continue
                s = 3 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    if self.conceptnet:
                        if connect_matrix[i, j - utt_feature.shape[1], k - s]:
                            cuu_source.append(k)
                            cuu_target.append(j)
                            edge_type.append(3)
                            edge_time.append(0)

                            cuu_source.append(k)
                            cuu_target.append(j + utt_feature.shape[1])
                            edge_type.append(3)
                            edge_time.append(0)
                    else:
                        cuu_source.append(k)
                        cuu_target.append(j)
                        edge_type.append(3)
                        edge_time.append(0)

                        cuu_source.append(k)
                        cuu_target.append(j + utt_feature.shape[1])
                        edge_type.append(3)
                        edge_time.append(0)

            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def cos_similarity(self, aggr, know):
        result = torch.nn.functional.cosine_similarity(
            aggr.unsqueeze(1).repeat(1, know.shape[1], 1), know, dim=2
        )
        return result

    def forward(self, inputs, mask, utt_pos_mask, comet_inputs, comet_mask):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.encoder(inputs, attention_mask=mask)[0]

        utt_xs = x * utt_pos_mask.unsqueeze(-1)
        utt_xs = torch.sum(utt_xs, dim=2) / (
            torch.sum(utt_pos_mask, dim=-1) + 1e-9
        ).unsqueeze(-1)

        if self.num_future_utts == 0:
            cuu_pos = utt_xs.shape[0] - 1
        else:
            cuu_pos = int((utt_xs.shape[0] - 1) / 2)
        cuu_know_pos_f = cuu_pos + utt_xs.shape[0]
        cuu_know_pos_b = cuu_know_pos_f + utt_xs.shape[0]
        cuu_input_know = utt_xs.shape[0] * 3 + comet_inputs.shape[2] * cuu_pos
        # x = utt_xs[cuu_pos,]

        # attn_queries = self.comet_attn_layer(utt_xs)
        # utt_attended_xs = self.scaled_dot_product_attention(attn_queries.transpose(0, 1), comet_inputs, comet_inputs)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        # latent_params = dict()
        # latent_params['utt'] = self.compute_latent_params(x, self.context2params['utt'])
        # latent_params['comet_utt'] = self.compute_latent_params(utt_attended_xs, self.context2params['comet_utt'])

        utt_xs = utt_xs.transpose(0, 1)
        utt_pos = (
            torch.LongTensor([i for i in range(utt_xs.shape[1])])
            .repeat(utt_xs.shape[0], 1)
            .to(self.device)
        )
        utt_xs = self.emb(utt_xs, utt_pos)

        # comet_utt_z = latent_params['comet_utt'].z
        comet_utt_f = torch.mean(
            torch.index_select(comet_inputs, 2, self.forward_pos), dim=-2
        )
        comet_utt_b = torch.mean(
            torch.index_select(comet_inputs, 2, self.backward_pos), dim=-2
        )
        # comet_utt_f = utt_xs.detach()
        # comet_utt_b = utt_xs.detach()
        # comet_utt_z = self.know_emb(comet_utt_z, utt_pos)
        # comet_utt_f = self.comet_utt_f.repeat(inputs.shape[0], 1, 1)
        # comet_utt_b = self.comet_utt_b.repeat(inputs.shape[0], 1, 1)

        if self.conv_name == "multidim_hgt":
            (
                node_feature,
                node_type,
                edge_index,
                edge_type,
                edge_time,
            ) = self.keepdim_make_graph(
                utt_xs, comet_mask, comet_utt_f, comet_utt_b, comet_inputs
            )
        else:
            """node_feature, node_type, edge_index, edge_type, edge_time = self.make_bipartite_graph(
                utt_xs, comet_mask, comet_utt_z, comet_inputs, cuu_pos)"""
            node_feature, node_type, edge_index, edge_type, edge_time = self.make_graph(
                utt_xs, comet_mask, comet_utt_f, comet_utt_b, comet_inputs
            )
        hgt_features = self.gnn(
            node_feature, node_type, edge_time, edge_index, edge_type
        )

        indices = torch.LongTensor(
            [
                cuu_pos + i * int(hgt_features.shape[0] / utt_xs.shape[0])
                for i in range(utt_xs.shape[0])
            ]
        )
        know_indices_f = torch.LongTensor(
            [
                cuu_know_pos_f + i * int(hgt_features.shape[0] / utt_xs.shape[0])
                for i in range(utt_xs.shape[0])
            ]
        )
        know_indices_b = torch.LongTensor(
            [
                cuu_know_pos_b + i * int(hgt_features.shape[0] / utt_xs.shape[0])
                for i in range(utt_xs.shape[0])
            ]
        )
        comet_know_indices = []
        for i in range(utt_xs.shape[0]):
            comet_know_indices += [
                cuu_input_know + i * int(hgt_features.shape[0] / utt_xs.shape[0]) + j
                for j in range(comet_inputs.shape[2])
            ]
        comet_know_indices = torch.LongTensor(comet_know_indices)
        tgt_features = torch.index_select(hgt_features, 0, indices.to(self.device))
        know_features_f = torch.index_select(
            hgt_features, 0, know_indices_f.to(self.device)
        )[:, : self.comet_hidden_size]
        know_features_b = torch.index_select(
            hgt_features, 0, know_indices_b.to(self.device)
        )[:, : self.comet_hidden_size]
        comet_features_output = torch.index_select(
            hgt_features, 0, comet_know_indices.to(self.device)
        )[:, : self.comet_hidden_size]
        comet_features_output = comet_features_output.reshape(
            utt_xs.shape[0], -1, comet_features_output.shape[-1]
        )
        f_cos = (
            self.cos_similarity(know_features_f, comet_features_output).detach().cpu()
        )
        b_cos = (
            self.cos_similarity(know_features_b, comet_features_output).detach().cpu()
        )

        latent_params = dict()
        """latent_params['utt'] = self.compute_latent_params(torch.cat([tgt_features, know_features_f, know_features_b], dim=1), self.context2params['utt'])
        utt_z = latent_params['utt'].z"""

        # cuu_comet_rep = comet_utt_z[:,cuu_pos,:]
        # cuu_comet_rep = torch.sum(comet_utt_z*comet_mask.unsqueeze(-1), dim=1) / torch.sum(comet_mask, dim=-1).unsqueeze(-1)
        # cuu_comet_rep = self.scaled_dot_product_attention(attn_queries[cuu_pos,], comet_utt_z, comet_utt_z, mask=comet_mask)
        # utt_z = torch.cat([utt_z, cuu_comet_rep], dim=-1)

        return (
            self.cu_utt_emo_prediction_layers(tgt_features),
            latent_params,
            f_cos,
            b_cos,
        )
        # return self.cu_utt_emo_prediction_layers(torch.cat([tgt_features, know_features], dim=-1)), latent_params


class CasualVIBERC(nn.Module):
    """The BHG model for CEE task."""

    def __init__(self, args, num_class):
        super(CasualVIBERC, self).__init__()
        self.device = args["device"]
        self.batch_size = args["BATCH_SIZE"]
        self.num_future_utts = args["num_future_utterances"]
        self.comet_hidden_size = args["COMET_HIDDEN_SIZE"]
        self.conceptnet = "conceptnet" in args["ROOT_DIR"]

        # Prepare the encoder and decoder for VAE.
        self.encoder = RobertaModel.from_pretrained(args["model_checkpoint"])
        self.tokenizer = RobertaTokenizer.from_pretrained(args["model_checkpoint"])
        self.config = AutoConfig.from_pretrained(args["model_checkpoint"])

        hidden_size = self.config.hidden_size
        self.conv_name = args["CONV_NAME"]

        if self.conv_name == "multidim_hgt":
            self.gnn = MultiDimGNN(
                h_dim=hidden_size,
                l_dim=self.comet_hidden_size,
                n_heads=8,
                n_layers=2,
                dropout=self.config.hidden_dropout_prob,
                num_types=4,
                num_relations=4,
                use_RTE=False,
            )
        else:
            self.gnn = GNN(
                conv_name=self.conv_name,
                in_dim=hidden_size,
                n_hid=hidden_size,
                n_heads=8,
                n_layers=2,
                dropout=self.config.hidden_dropout_prob,
                num_types=4,
                num_relations=4,
                use_RTE=False,
            )

        self.context2params = nn.ModuleDict()

        """The variational modules for target utterance representations."""
        """utt_layer = nn.Linear(
            hidden_size+2*comet_hidden_size, 2 * hidden_size)
        self.context2params["utt"] = utt_layer"""

        # self.comet_utt_layer = nn.Linear(
        #    hidden_size, comet_hidden_size)

        """The variational modules for comet utterance-level knowledge representations."""
        """self.comet_attn_layer = nn.Linear(hidden_size, comet_hidden_size)
        comet_utt_layer = nn.Linear(
            comet_hidden_size, 2 * comet_hidden_size)
        self.context2params["comet_utt"] = comet_utt_layer"""

        """self.comet_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(comet_hidden_size, comet_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(comet_hidden_size, num_class))"""

        self.emb = RelTemporalEncoding(hidden_size)
        # self.know_emb = RelTemporalEncoding(comet_hidden_size)
        self.forward_pos = torch.LongTensor([i for i in range(5)]).to(self.device)
        self.backward_pos = torch.LongTensor([i for i in range(5, 7)]).to(self.device)

        self.cu_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_class),
        )
        self.cause_emo_prediction_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        :param Q: [B, WIN, D]
        :param K: [B, WIN, 9, D]
        :param V: [B, WIN, 9, D]
        :param mask: [B, WIN]
        :return: Attended results.
        """
        attn_weights = torch.matmul(K, Q.unsqueeze(-1)).squeeze(-1) / torch.sqrt(
            torch.tensor(Q.shape[-1])
        )
        if mask is not None:
            mask = (1.0 - mask) * -100000.0
            attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights.unsqueeze(-2), V).squeeze(-2)

    def compute_latent_params(self, context, layer, mode="train"):
        """Estimate the latent parameters."""
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        params = layer(context)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.tanh(logvar)
        if mode == "train":
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            z = mu

        return Params(z, mu, logvar)

    def keepdim_make_graph(
        self, utt_feature, agg_feature_forward, agg_feature_back, comet_feature
    ):
        """
                :param utt_feature: [B, WIN, D1]
                :param utt_mask: [B, WIN]
                :param agg_feature: [B, WIN, D2]
                :param comet_feature: [B, WIN, 9, D2]
                :return:
        """
        if self.conceptnet:
            connect_matrix = torch.sum(comet_feature, dim=-1) != 0
        agg_feature_forward = torch.cat(
            [
                agg_feature_forward,
                torch.zeros(
                    [
                        agg_feature_forward.shape[0],
                        agg_feature_forward.shape[1],
                        utt_feature.shape[-1] - agg_feature_forward.shape[2],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        agg_feature_back = torch.cat(
            [
                agg_feature_back,
                torch.zeros(
                    [
                        agg_feature_back.shape[0],
                        agg_feature_back.shape[1],
                        utt_feature.shape[-1] - agg_feature_back.shape[2],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        comet_feature = torch.cat(
            [
                comet_feature,
                torch.zeros(
                    [
                        comet_feature.shape[0],
                        comet_feature.shape[1],
                        comet_feature.shape[2],
                        utt_feature.shape[-1] - comet_feature.shape[3],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature_forward,
                agg_feature_back,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * utt_feature.shape[1]
                + [3] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            for j in range(utt_feature.shape[1]):
                cuu_source.append(j)
                cuu_target.append(j + utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

                cuu_source.append(j)
                cuu_target.append(j + 2 * utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                for k in range(j - utt_feature.shape[1], utt_feature.shape[1]):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(2)

            for j in range(2 * utt_feature.shape[1], 3 * utt_feature.shape[1]):
                for k in range(0, j - 2 * utt_feature.shape[1] + 1):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(2)
                    edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                s = 3 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    if self.conceptnet:
                        if connect_matrix[i, j - utt_feature.shape[1], k - s]:
                            cuu_source.append(k)
                            cuu_target.append(j)
                            edge_type.append(3)
                            edge_time.append(0)

                            cuu_source.append(k)
                            cuu_target.append(j + utt_feature.shape[1])
                            edge_type.append(3)
                            edge_time.append(0)
                    else:
                        cuu_source.append(k)
                        cuu_target.append(j)
                        edge_type.append(3)
                        edge_time.append(0)

                        cuu_source.append(k)
                        cuu_target.append(j + utt_feature.shape[1])
                        edge_type.append(3)
                        edge_time.append(0)

            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def make_graph(
        self, utt_feature, agg_feature_forward, agg_feature_back, comet_feature
    ):
        """
        :param utt_feature: [B, WIN, D]
        :param utt_mask: [B, WIN]
        :param agg_feature_forward: [B, WIN, D]
        :param agg_feature_back: [B, WIN, D]
        :param comet_feature: [B, WIN, 9, D]
        :return:
        """
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature_forward,
                agg_feature_back,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * utt_feature.shape[1]
                + [3] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            for j in range(utt_feature.shape[1]):
                cuu_source.append(j)
                cuu_target.append(j + utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

                cuu_source.append(j)
                cuu_target.append(j + 2 * utt_feature.shape[1])
                edge_type.append(0)
                edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                for k in range(j - utt_feature.shape[1], utt_feature.shape[1]):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(2)

            for j in range(2 * utt_feature.shape[1], 3 * utt_feature.shape[1]):
                for k in range(0, j - 2 * utt_feature.shape[1] + 1):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(2)
                    edge_time.append(2)

            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                s = 3 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    cuu_source.append(k)
                    cuu_target.append(j)
                    edge_type.append(3)
                    edge_time.append(0)

                    cuu_source.append(k)
                    cuu_target.append(j + utt_feature.shape[1])
                    edge_type.append(3)
                    edge_time.append(0)

            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def forward(self, inputs, mask, utt_pos_mask, comet_inputs):
        """
        :param inputs: The input of PLM. Dim: [B, seq_len]
        :param mask: The mask for input x. Dim: [B, seq_len]
        """
        x = self.encoder(inputs.unsqueeze(0), attention_mask=mask.unsqueeze(0))[
            0
        ].squeeze(0)

        utt_xs = x * utt_pos_mask.unsqueeze(-1)
        utt_xs = torch.sum(utt_xs, dim=1) / (torch.sum(utt_pos_mask, dim=-1)).unsqueeze(
            -1
        )

        # attn_queries = self.comet_attn_layer(utt_xs)
        # utt_attended_xs = self.scaled_dot_product_attention(attn_queries.transpose(0, 1), comet_inputs, comet_inputs)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        # latent_params = dict()
        # latent_params['utt'] = self.compute_latent_params(x, self.context2params['utt'])
        # latent_params['comet_utt'] = self.compute_latent_params(utt_attended_xs, self.context2params['comet_utt'])

        """utt_xs = utt_xs.transpose(0, 1)
        utt_pos = torch.LongTensor(
            [i for i in range(utt_xs.shape[1])]).repeat(utt_xs.shape[0], 1).to(self.device)"""
        utt_pos = torch.LongTensor([i for i in range(utt_xs.shape[0])]).to(self.device)
        utt_xs = self.emb(utt_xs.unsqueeze(0), utt_pos.unsqueeze(0))

        # comet_utt_z = latent_params['comet_utt'].z
        comet_utt_f = torch.mean(
            torch.index_select(comet_inputs, 1, self.forward_pos), dim=-2
        )
        comet_utt_b = torch.mean(
            torch.index_select(comet_inputs, 1, self.backward_pos), dim=-2
        )
        # comet_utt_z = self.know_emb(comet_utt_z, utt_pos)

        if self.conv_name == "multidim_hgt":
            (
                node_feature,
                node_type,
                edge_index,
                edge_type,
                edge_time,
            ) = self.keepdim_make_graph(
                utt_xs,
                comet_utt_f.unsqueeze(0),
                comet_utt_b.unsqueeze(0),
                comet_inputs.unsqueeze(0),
            )
        else:
            """node_feature, node_type, edge_index, edge_type, edge_time = self.make_bipartite_graph(
                utt_xs, comet_mask, comet_utt_z, comet_inputs, cuu_pos)"""
            node_feature, node_type, edge_index, edge_type, edge_time = self.make_graph(
                utt_xs,
                comet_utt_f.unsqueeze(0),
                comet_utt_b.unsqueeze(0),
                comet_inputs.unsqueeze(0),
            )
        hgt_features = self.gnn(
            node_feature, node_type, edge_time, edge_index, edge_type
        )

        """indices = torch.LongTensor(
            [cuu_pos + i * int(hgt_features.shape[0] / utt_xs.shape[0]) for i in range(utt_xs.shape[0])])
        know_indices_f = torch.LongTensor(
            [cuu_know_pos_f + i * int(hgt_features.shape[0] / utt_xs.shape[0]) for i in range(utt_xs.shape[0])])
        know_indices_b = torch.LongTensor(
            [cuu_know_pos_b + i * int(hgt_features.shape[0] / utt_xs.shape[0]) for i in range(utt_xs.shape[0])])"""
        # tgt_features = torch.index_select(hgt_features, 0, indices.to(self.device))
        tgt_features = hgt_features[: utt_xs.shape[1]]
        # print(tgt_features.shape)
        cause_features = torch.cat(
            [
                tgt_features,
                tgt_features[-1].unsqueeze(0).repeat(tgt_features.shape[0], 1),
            ],
            dim=-1,
        )
        """know_features_f = torch.index_select(hgt_features, 0, know_indices_f.to(self.device))
        know_features_b = torch.index_select(hgt_features, 0, know_indices_b.to(self.device))"""

        latent_params = dict()
        """latent_params['utt'] = self.compute_latent_params(torch.cat([tgt_features, know_features_f, know_features_b], dim=1), self.context2params['utt'])
        utt_z = latent_params['utt'].z"""

        # cuu_comet_rep = comet_utt_z[:,cuu_pos,:]
        # cuu_comet_rep = torch.sum(comet_utt_z*comet_mask.unsqueeze(-1), dim=1) / torch.sum(comet_mask, dim=-1).unsqueeze(-1)
        # cuu_comet_rep = self.scaled_dot_product_attention(attn_queries[cuu_pos,], comet_utt_z, comet_utt_z, mask=comet_mask)
        # utt_z = torch.cat([utt_z, cuu_comet_rep], dim=-1)

        return (
            self.cu_utt_emo_prediction_layers(tgt_features),
            self.cause_emo_prediction_layers(cause_features).transpose(1, 0).squeeze(0),
            latent_params,
        )
        # return self.cu_utt_emo_prediction_layers(torch.cat([tgt_features, know_features], dim=-1)), latent_params


class GraphERC(nn.Module):
    def __init__(self, args, num_class, comet_hidden_size=768):
        super(GraphERC, self).__init__()
        self.device = args["device"]
        self.batch_size = args["BATCH_SIZE"]
        self.num_future_utts = args["num_future_utterances"]
        self.comet_hidden_size = comet_hidden_size

        # Prepare the encoder and decoder for VAE.
        self.encoder = RobertaModel.from_pretrained(args["model_checkpoint"])
        self.tokenizer = RobertaTokenizer.from_pretrained(args["model_checkpoint"])
        self.config = AutoConfig.from_pretrained(args["model_checkpoint"])

        hidden_size = self.config.hidden_size
        self.conv_name = "multidim_hgt"

        if self.conv_name == "multidim_hgt":
            self.gnn = MultiDimGNN(
                h_dim=hidden_size,
                l_dim=comet_hidden_size,
                n_heads=8,
                n_layers=3,
                dropout=self.config.hidden_dropout_prob,
                num_types=3,
                num_relations=4,
            )
        else:
            self.gnn = GNN(
                conv_name=self.conv_name,
                in_dim=comet_hidden_size,
                n_hid=comet_hidden_size,
                n_heads=8,
                n_layers=3,
                dropout=self.config.hidden_dropout_prob,
                num_types=3,
                num_relations=3,
            )

        self.context2params = nn.ModuleDict()

        """The variational modules for target utterance representations."""
        utt_layer = nn.Linear(hidden_size, 2 * hidden_size)
        self.context2params["utt"] = utt_layer

        self.comet_utt_layer = nn.Linear(hidden_size, comet_hidden_size)

        self.emb = RelTemporalEncoding(hidden_size)

        """The variational modules for comet utterance-level knowledge representations."""
        """self.comet_attn_layer = nn.Linear(hidden_size, comet_hidden_size)
        comet_utt_layer = nn.Linear(
            comet_hidden_size, 2 * comet_hidden_size)
        self.context2params["comet_utt"] = comet_utt_layer"""

        """self.comet_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(comet_hidden_size, comet_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(comet_hidden_size, num_class))"""

        self.cu_utt_emo_prediction_layers = nn.Sequential(
            nn.Linear(hidden_size + comet_hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_class),
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        :param Q: [B, WIN, D]
        :param K: [B, WIN, 9, D]
        :param V: [B, WIN, 9, D]
        :param mask: [B, WIN]
        :return: Attended results.
        """
        attn_weights = torch.matmul(K, Q.unsqueeze(-1)).squeeze(-1) / torch.sqrt(
            torch.tensor(Q.shape[-1])
        )
        if mask is not None:
            mask = (1.0 - mask) * -100000.0
            attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights.unsqueeze(-2), V).squeeze(-2)

    def compute_latent_params(self, context, layer, mode="train"):
        """Estimate the latent parameters."""
        Params = namedtuple("Params", ["z", "mu", "logvar"])
        params = layer(context)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.tanh(logvar)
        if mode == "train":
            z = mu + torch.randn_like(logvar) * torch.exp(logvar)
        else:
            z = mu

        return Params(z, mu, logvar)

    def kepdim_make_graph(
        self, utt_feature, utt_mask, agg_feature, comet_feature, cuu_pos
    ):
        """
                :param utt_feature: [B, WIN, D1]
                :param utt_mask: [B, WIN]
                :param agg_feature: [B, WIN, D2]
                :param comet_feature: [B, WIN, 9, D2]
                :return:
        """
        agg_feature = torch.cat(
            [
                agg_feature,
                torch.zeros(
                    [
                        agg_feature.shape[0],
                        agg_feature.shape[1],
                        utt_feature.shape[-1] - agg_feature.shape[2],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        comet_feature = torch.cat(
            [
                comet_feature,
                torch.zeros(
                    [
                        comet_feature.shape[0],
                        comet_feature.shape[1],
                        comet_feature.shape[2],
                        utt_feature.shape[-1] - comet_feature.shape[3],
                    ],
                    dtype=torch.float,
                ).to(self.device),
            ],
            dim=-1,
        )
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            pos = (utt_mask[i,] == 1).nonzero()
            for j in range(pos.shape[0] - 1):
                cuu_source.append(int(pos[j]))
                cuu_target.append(int(pos[j + 1]))
                edge_type.append(0)
                edge_time.append(0)
            for j in range(pos.shape[0]):
                if int(pos[j]) == cuu_pos - 1:
                    continue
                cuu_source.append(int(pos[j]))
                cuu_target.append(cuu_pos)
                edge_type.append(0)
                edge_time.append(0)
            for j in range(utt_feature.shape[1]):
                for k in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(1)

                    cuu_source.append(k)
                    cuu_target.append(j)
                    edge_type.append(2)
                    edge_time.append(3)
            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                s = 2 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    cuu_source.append(k)
                    cuu_target.append(j)
                    edge_type.append(3)
                    edge_time.append(2)
            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def make_graph(self, utt_feature, utt_mask, agg_feature, comet_feature, cuu_pos):
        """
        :param utt_feature: [B, WIN, D]
        :param utt_mask: [B, WIN]
        :param agg_feature: [B, WIN, D]
        :param comet_feature: [B, WIN, 9, D]
        :return:
        """
        node_feature = torch.cat(
            [
                utt_feature,
                agg_feature,
                comet_feature.view(comet_feature.shape[0], -1, comet_feature.shape[-1]),
            ],
            dim=1,
        ).view(-1, comet_feature.shape[-1])

        node_type = torch.LongTensor(
            (
                [0] * utt_feature.shape[1]
                + [1] * utt_feature.shape[1]
                + [2] * (utt_feature.shape[1] * comet_feature.shape[-2])
            )
            * utt_feature.shape[0]
        )

        edge_index = [[], []]
        edge_type = []
        edge_time = []
        for i in range(utt_feature.shape[0]):
            cuu_source = []
            cuu_target = []
            pos = (utt_mask[i,] == 1).nonzero()
            for j in range(pos.shape[0]):
                cuu_source.append(int(pos[j]))
                cuu_target.append(cuu_pos)
                edge_type.append(0)
                edge_time.append(3)
            for j in range(utt_feature.shape[1]):
                for k in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                    cuu_source.append(j)
                    cuu_target.append(k)
                    edge_type.append(1)
                    edge_time.append(2)

                    cuu_source.append(k)
                    cuu_target.append(j)
                    edge_type.append(2)
                    edge_time.append(1)
            for j in range(utt_feature.shape[1], 2 * utt_feature.shape[1]):
                s = 2 * utt_feature.shape[1] + comet_feature.shape[-2] * (
                    j - utt_feature.shape[1]
                )
                for k in range(s, s + comet_feature.shape[-2]):
                    cuu_source.append(k)
                    cuu_target.append(j)
                    edge_type.append(3)
                    edge_time.append(0)
            edge_index[0] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_source
            ]
            edge_index[1] += [
                j + i * int(node_feature.shape[0] / utt_feature.shape[0])
                for j in cuu_target
            ]
        edge_index = torch.LongTensor(edge_index)
        edge_type = torch.LongTensor(edge_type)
        edge_time = torch.LongTensor(edge_time)

        return (
            node_feature.to(self.device),
            node_type.to(self.device),
            edge_index.to(self.device),
            edge_type.to(self.device),
            edge_time.to(self.device),
        )

    def forward(self, inputs, mask, comet_inputs, comet_mask):
        """
        :param inputs: The input of PLM. Dim: [B, WIN, seq_len]
        :param mask: The mask for input x. Dim: [B, WIN, seq_len]
        """
        x = self.encoder(inputs, attention_mask=mask)[0][:, 0, :]

        utt_pos = torch.LongTensor(
            [i for i in range(comet_mask.shape[-1])]
            * int(x.shape[0] / comet_mask.shape[-1])
        ).to(self.device)
        x = self.emb(x, utt_pos)
        x = x.reshape(-1, comet_mask.shape[-1], x.shape[-1])

        if self.num_future_utts == 0:
            cuu_pos = x.shape[1] - 1
        else:
            cuu_pos = int((x.shape[1] - 1) / 2)
        cuu_know_pos = cuu_pos + x.shape[1]
        # x = utt_xs[cuu_pos,]

        # attn_queries = self.comet_attn_layer(utt_xs)
        # utt_attended_xs = self.scaled_dot_product_attention(attn_queries.transpose(0, 1), comet_inputs, comet_inputs)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = dict()
        # latent_params['utt'] = self.compute_latent_params(x, self.context2params['utt'])
        # latent_params['comet_utt'] = self.compute_latent_params(utt_attended_xs, self.context2params['comet_utt'])

        # comet_utt_z = latent_params['comet_utt'].z
        comet_utt_z = torch.mean(comet_inputs, dim=-2)

        if self.conv_name == "multidim_hgt":
            (
                node_feature,
                node_type,
                edge_index,
                edge_type,
                edge_time,
            ) = self.kepdim_make_graph(
                x, comet_mask, comet_utt_z, comet_inputs, cuu_pos
            )
        else:
            node_feature, node_type, edge_index, edge_type, edge_time = self.make_graph(
                self.comet_utt_layer(x), comet_mask, comet_utt_z, comet_inputs, cuu_pos
            )
        hgt_features = self.gnn(
            node_feature, node_type, edge_time, edge_index, edge_type
        )

        indices = torch.LongTensor(
            [
                cuu_pos + i * int(hgt_features.shape[0] / x.shape[0])
                for i in range(x.shape[0])
            ]
        )
        know_indices = torch.LongTensor(
            [
                cuu_know_pos + i * int(hgt_features.shape[0] / x.shape[0])
                for i in range(x.shape[0])
            ]
        )
        tgt_features = torch.index_select(hgt_features, 0, indices.to(self.device))
        know_features = torch.index_select(
            hgt_features, 0, know_indices.to(self.device)
        )[:, : self.comet_hidden_size]

        # cuu_comet_rep = comet_utt_z[:,cuu_pos,:]
        # cuu_comet_rep = torch.sum(comet_utt_z*comet_mask.unsqueeze(-1), dim=1) / torch.sum(comet_mask, dim=-1).unsqueeze(-1)
        # cuu_comet_rep = self.scaled_dot_product_attention(attn_queries[cuu_pos,], comet_utt_z, comet_utt_z, mask=comet_mask)
        # utt_z = torch.cat([utt_z, cuu_comet_rep], dim=-1)

        # return self.cu_utt_emo_prediction_layers(tgt_features), latent_params
        return (
            self.cu_utt_emo_prediction_layers(
                torch.cat([tgt_features, know_features], dim=-1)
            ),
            latent_params,
        )
