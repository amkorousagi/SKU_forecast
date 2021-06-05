import torch.nn as nn
import torch as my_torch


class ForecastingModel(nn.Module):
    # feat_dim 8 or 4
    def __init__(self, hist_len, pred_len, prod_num_embeddings, feat_dim = 8):
        super().__init__()
        self.hist_len = hist_len

        self.order_hist_layer = nn.Linear(hist_len, feat_dim * 3)
        self.price_hist_layer = nn.Linear(hist_len, 1)
        self.price_pred_layer = nn.Linear(pred_len, 1)
        self.prod_feat_mrc_layer = nn.Linear(1,1)
        self.prod_feat_cat_layer = nn.Linear(4,1)
        '''
        self.prod_feat_layers = nn.ModuleList([
            nn.Embedding(n_e+2, feat_dim, padding_idx=0) for n_e in prod_num_embeddings
        ])
        '''
        # method 1 : feat_dim = 16 , hist_len=350, <feat_dim*5 + feat_dim*2> : mrc 과 cat 에도 비중을 주자
        # method 2 : LSTM 이전의 기억...
        # nn.LeakyReLU() # 200 ,100 100,7 
        self.pred_layer = nn.Sequential(
            nn.Linear( feat_dim * 3 + 2, feat_dim *3),
            nn.LeakyReLU(0.1),
            nn.Linear( feat_dim*3, feat_dim *2),
            nn.LeakyReLU(0.1),
            nn.Linear( feat_dim *2, feat_dim ),
            nn.LeakyReLU(0.1),
            nn.Linear( feat_dim  , pred_len)
        )

    def forward(self, train_order_hist, train_price_hist, test_price_hist, prod_feat):
        # print("my prod feat")
        # print(my_torch.tensor(prod_feat,dtype=int))
        # print("prod : " + str(prod_feat.shape))
        # print(str(prod_feat[:,0].view(len(prod_feat[:,0]),1]).shape))
        # print(str(prod_feat[:,0].unsqueeze(0).T.shape))
        
        order_hist_emb = self.order_hist_layer(train_order_hist)
        price_hist_emb = self.price_hist_layer(train_price_hist)
        price_pred_emb = self.price_pred_layer(test_price_hist)
        prod_feat_mrc_emb = self.prod_feat_mrc_layer(prod_feat[:,0].unsqueeze(0).T)
        prod_feat_cat_emb = self.prod_feat_cat_layer(prod_feat[:,1:5])

        # append 64 col + 64 col + 64 col + 5 col -> N* 207
        # np.append 에 대해 조금 알아봐서 이렇게 만들면 됨.
        emb_all = order_hist_emb
        emb_all = my_torch.cat([emb_all, price_hist_emb], dim=1)
        emb_all = my_torch.cat([emb_all, price_pred_emb], dim=1)
        # emb_all = my_torch.cat([emb_all, prod_feat_mrc_emb], dim=1)
        # emb_all = my_torch.cat([emb_all, prod_feat_cat_emb], dim=1)

        pred = self.pred_layer(emb_all)
        return pred

    def get_hist_len(self):
        return self.hist_len
