
import torch.nn as nn


class ForecastingModel(nn.Module):
    def __init__(self, hist_len, pred_len, prod_num_embeddings, feat_dim=64):
        super().__init__()
        self.hist_len = hist_len

        self.order_hist_layer = nn.Linear(hist_len, feat_dim)
        self.price_hist_layer = nn.Linear(hist_len, feat_dim)
        self.price_pred_layer = nn.Linear(pred_len, feat_dim)
        self.prod_feat_layers = nn.ModuleList([
            nn.Embedding(n_e+2, feat_dim, padding_idx=0) for n_e in prod_num_embeddings
        ])
        self.pred_layer = nn.Sequential(nn.ReLU(), nn.Linear(feat_dim, pred_len))

    def forward(self, train_order_hist, train_price_hist, test_price_hist, prod_feat):
        order_hist_emb = self.order_hist_layer(train_order_hist)
        price_hist_emb = self.price_hist_layer(train_price_hist)
        price_pred_emb = self.price_pred_layer(test_price_hist)

        emb_all = order_hist_emb + price_hist_emb + price_pred_emb
        for i, model in enumerate(self.prod_feat_layers):
            emb_all += model(prod_feat[:, i]+1)

        pred = self.pred_layer(emb_all)
        return pred

    def get_hist_len(self):
        return self.hist_len
