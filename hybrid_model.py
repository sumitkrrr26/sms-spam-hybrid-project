from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel


class GraphConvolution(nn.Module):
    """Simple normalized graph convolution over dense adjacency matrices."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_features: torch.Tensor,
        adjacency: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Explicitly remove padding edges and add self-loops on real tokens.
        mask = token_mask.float()
        adjacency = adjacency.float()
        adjacency = adjacency * mask.unsqueeze(1) * mask.unsqueeze(2)
        eye = torch.diag_embed(mask)
        adjacency = torch.maximum(adjacency, eye)
        degree = adjacency.sum(dim=-1).clamp(min=1.0)
        inv_sqrt = degree.rsqrt()
        normalized = inv_sqrt.unsqueeze(2) * adjacency * inv_sqrt.unsqueeze(1)
        aggregated = torch.bmm(normalized, token_features)
        output = torch.nn.functional.gelu(self.linear(aggregated))
        output = self.dropout(output)
        return output * mask.unsqueeze(-1)


class HybridSpamNet(nn.Module):
    """
    BERT semantic branch + Word2Vec/BiLSTM sequence branch + 3 GCN branches.
    Branch summaries are stacked and combined with multi-head self-attention.
    The final layer returns a logit; apply sigmoid to obtain spam probability.
    """

    def __init__(
        self,
        embedding_matrix,
        bert_name: str = "bert-base-uncased",
        dropout: float = 0.25,
        fusion_dim: int = 128,
        lstm_hidden: int = 64,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = int(self.bert.config.hidden_size)

        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        embedding_tensor = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        self.word_embedding = nn.Embedding.from_pretrained(
            embedding_tensor, freeze=False, padding_idx=0
        )
        word_dim = int(embedding_tensor.shape[1])
        self.bilstm = nn.LSTM(
            input_size=word_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        lstm_dim = lstm_hidden * 2

        self.gcn_cooccurrence = GraphConvolution(bert_dim, fusion_dim, dropout)
        self.gcn_heterogeneous = GraphConvolution(bert_dim, fusion_dim, dropout)
        self.gcn_syntactic = GraphConvolution(bert_dim, fusion_dim, dropout)

        self.bert_projection = nn.Sequential(
            nn.Linear(bert_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm_projection = nn.Sequential(
            nn.Linear(lstm_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_norm = nn.LayerNorm(fusion_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
        )

    @staticmethod
    def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor,
        graphs: torch.Tensor,
    ) -> torch.Tensor:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        bert_summary = self.masked_mean(bert_output, attention_mask)
        bert_branch = self.bert_projection(bert_summary)

        word_vectors = self.word_embedding(word_ids)
        word_lengths = (word_ids != 0).sum(dim=1).clamp(min=1).to("cpu")
        packed = pack_padded_sequence(
            word_vectors,
            word_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        lstm_sequence, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=word_ids.shape[1]
        )
        word_mask = (word_ids != 0).long()
        lstm_summary = self.masked_mean(lstm_sequence, word_mask)
        lstm_branch = self.lstm_projection(lstm_summary)

        graph_branches = []
        graph_layers = (
            self.gcn_cooccurrence,
            self.gcn_heterogeneous,
            self.gcn_syntactic,
        )
        for branch_index, layer in enumerate(graph_layers):
            # Graph construction marks real text-subword nodes on the diagonal.
            # This excludes BERT's [CLS]/[SEP] special tokens and padding from
            # graph pooling, while the separate BERT summary still uses attention_mask.
            graph_mask = torch.diagonal(
                graphs[:, branch_index], dim1=1, dim2=2
            ).long()
            nodes = layer(bert_output, graphs[:, branch_index], graph_mask)
            graph_branches.append(self.masked_mean(nodes, graph_mask))

        # Five branch vectors become five "tokens" for self-attention:
        # BERT, BiLSTM, co-occurrence GCN, heterogeneous GCN, syntactic GCN.
        branch_tokens = torch.stack(
            [bert_branch, lstm_branch, *graph_branches], dim=1
        )
        branch_tokens = self.branch_norm(branch_tokens)
        attended, _ = self.self_attention(
            branch_tokens, branch_tokens, branch_tokens, need_weights=False
        )
        fused = self.dropout(attended.mean(dim=1))
        return self.classifier(fused).squeeze(-1)
