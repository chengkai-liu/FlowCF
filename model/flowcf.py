from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import typing


class FlowModel(nn.Module):
    def __init__(
        self,
        dims: typing.List,
        time_emb_size: int,
        time_type="cat",
        act_func="tanh",
        norm=False,
        init_dropout=0
    ):
        super(FlowModel, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = time_emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            # Concatenate timestep embedding with input
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )

        self.mlp_layers = MLPLayers(
            layers=self.dims, dropout=0.1, activation=act_func, last_activation=False
        )
        self.init_dropout = nn.Dropout(init_dropout)
        self.apply(xavier_normal_initialization)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.init_dropout(x)
        h = torch.cat([x, emb], dim=-1)
        output = self.mlp_layers(h)
        return output


class FlowCF(GeneralRecommender, AutoEncoderMixin):
    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(FlowCF, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

        self.n_steps = config["n_steps"]
        self.s_steps = config["s_steps"]
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)

        self.time_emb_size = config["time_embedding_size"]
        dims = [self.n_items] + config["dims_mlp"] + [self.n_items]

        self.flow_model = FlowModel(
            dims=dims,
            time_emb_size=self.time_emb_size,
            init_dropout=0
        )

        self.item_frequencies = self.get_item_frequencies()

    def get_item_frequencies(self):
        item_counts = torch.zeros(self.n_items, device=self.device)
        for user_id in range(self.n_users):
            user_item_ids = self.history_item_id[user_id]
            item_counts[user_item_ids] += 1

        item_frequencies = item_counts / self.n_users
        return item_frequencies

    def forward(self, x, t):
        return self.flow_model(x, t)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user)

        # Randomly sample steps
        steps = torch.randint(0, self.n_steps, (x1.size(0),), device=self.device)
        t = self.time_steps.to(x1.device)[steps].unsqueeze(1)

        # Sample x0 from behavior-guided prior
        x0 = torch.bernoulli(self.item_frequencies.expand(
            x1.size(0), -1)).to(x1.device)
        random_mask = torch.rand_like(x1, dtype=torch.float32) <= t
        # Interpolation between x0 and x1
        xt = torch.where(random_mask, x1, x0)

        model_output = self.forward(xt, t.squeeze(-1))
        loss = mean_flat((x1 - model_output) ** 2)
        loss = loss.mean()

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        X_bar = self.get_rating_matrix(user)
        Xt = X_bar

        for i_t in range(self.n_steps-self.s_steps, self.n_steps):
            t = self.time_steps[i_t].repeat(Xt.shape[0], 1).to(X_bar.device)
            X1_hat = self.forward(Xt, t.squeeze(-1))
            if i_t == self.n_steps-1:
                break
            t_next = self.time_steps[i_t + 1].repeat(Xt.shape[0], 1).to(X_bar.device)
            v = (X1_hat - Xt) / (1 - t)
            Xt_pos_probs = Xt + v * (t_next - t)
            Xt_neg_probs = 1 - Xt_pos_probs
            Xt_probs = torch.stack([Xt_neg_probs, Xt_pos_probs], dim=-1)
            # Sample from Xt_probs, considering the unique nature of collaborative filtering, we only allow 0 -> 1
            Xt = Xt_probs.argmax(dim=-1)
            # Preverve the observed interactions
            Xt = torch.logical_or(X_bar.to(torch.bool), Xt.to(torch.bool))
            Xt = Xt.float()

        return X1_hat

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t[:, item]
        return scores


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(
        timesteps.device
    ) * 2 * math.pi  # shape (dim//2,)
    args = timesteps[:, None].float() * freqs[None]  # (N, dim//2)
    embedding = torch.cat(
        [torch.cos(args), torch.sin(args)], dim=-1)  # (N, (dim//2)*2)
    if dim % 2:
        # zero pad in the last dimension to ensure shape (N, dim)
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
