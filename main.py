
import os
import argparse
import logging

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import nsml

from model import ForecastingModel
from config import FORECAST_LEN, USECOLS


def bind_model(model, device):
    def save(dir_name):
        params_fname = os.path.join(dir_name, 'params.pkl')
        torch.save(model.state_dict(), params_fname)

    def load(dir_name):
        params_fname = os.path.join(dir_name, 'params.pkl')
        if device.type == 'cpu':
            state = torch.load(params_fname, map_location={'cuda:0': 'cpu'})
        else:
            state = torch.load(params_fname)
        model.load_state_dict(state)

    def infer(
            train_order_hist: np.ndarray, train_price_hist: np.ndarray,
            test_price_hist: np.ndarray, prod_feat: np.ndarray
    ):
        """
        :params
            train_order_hist (T_H, N): The order quantity of N=500 products for the last T_H=735 days.
            train_price_hist (T_H, N): The price of N=500 products for the last T_H=735 days.
            test_price_hist (T_F, N): The price of N=500 products for the next T_F=7 days.
            prod_feat (N, F): The F=5 categorical features of each product.
        :return infer_result (T_F, N): The predicted order quantity of N=500 products for the next T_F=7 days.
        """
        model.eval()
        hist_len = model.get_hist_len()

        train_order_hist = torch.tensor(train_order_hist[-hist_len:].T, dtype=torch.float32, device=device)
        train_price_hist = torch.tensor(train_price_hist[-hist_len:].T, dtype=torch.float32, device=device)
        test_price_hist = torch.tensor(test_price_hist.T, dtype=torch.float32, device=device)
        prod_feat = torch.tensor(prod_feat, dtype=torch.long, device=device)

        prediction_tensor = model(train_order_hist, train_price_hist, test_price_hist, prod_feat)
        infer_result = prediction_tensor.detach().cpu().numpy().T
        return infer_result

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def main():
    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    # EXAMPLE: You can change as you want
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    config = args.parse_args()

    # Settings
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    )

    # Build and bind model
    logging.info('Build model...')
    prod_num_embeddings = [215, 49, 393, 2058, 2504]  # np.max(prod_feat, 0).astype(int).tolist()
    hist_len = FORECAST_LEN * 3
    pred_len = FORECAST_LEN
    model = ForecastingModel(hist_len, pred_len, prod_num_embeddings)
    bind_model(model, device)

    # DONOTCHANGE: They are reserved for nsml
    # Warning: Do not load data before the following code!
    if config.pause:
        nsml.paused(scope=locals())

    # Load data - You can freely use below data to train your own model !!!
    logging.info('Load data...')
    train_dataset_path = os.path.join(nsml.DATASET_PATH, 'train', 'train_data')
    order_hists = sp.load_npz(os.path.join(train_dataset_path, 'order_hist.npz'))
    price_hists = sp.load_npz(os.path.join(train_dataset_path, 'price_hist.npz'))
    prod_feat = pd.read_csv(
        os.path.join(train_dataset_path, 'prod_feat.csv'),
        usecols=USECOLS, dtype=int
    ).values

    # Train
    logging.info('Train model...')
    batch_size = 1024
    max_iter = 10000
    lr = 1e-3
    t_len, n_prod = order_hists.shape

    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for n_iter in range(1, max_iter+1):
        # Sample data
        t_sample = np.random.randint(hist_len, t_len-pred_len)
        nonzero_cnts = np.asarray((order_hists[t_sample:t_sample+pred_len] > 0).sum(0))[0]
        sample_prob = nonzero_cnts / np.sum(nonzero_cnts)
        p_sample = np.random.choice(range(n_prod), batch_size, p=sample_prob)
        order_hist_sample = order_hists[:, p_sample]
        price_hist_sample = price_hists[:, p_sample]

        # Convert to Tensor
        prod_feat_sample = torch.tensor(prod_feat[p_sample], dtype=torch.long)
        train_order_hist = csr_to_tensor(order_hist_sample[t_sample-hist_len:t_sample], device)
        train_price_hist = csr_to_tensor(price_hist_sample[t_sample-hist_len:t_sample], device)
        test_order_hist = csr_to_tensor(order_hist_sample[t_sample:t_sample+pred_len], device)
        test_price_hist = csr_to_tensor(price_hist_sample[t_sample:t_sample+pred_len], device)

        # Optimize
        optim.zero_grad()
        out = model(train_order_hist, train_price_hist, test_price_hist, prod_feat_sample)
        err = out - test_order_hist
        denom = 1 + test_order_hist.to_dense().mean(1, keepdim=True)
        loss = torch.mean((err / denom)**2)
        loss.backward()
        optim.step()

        # Report evaluation metrics
        mae = torch.mean(torch.abs(out - test_order_hist))
        wape = torch.mean(torch.abs(out - test_order_hist) / test_order_hist.to_dense().mean(1, keepdim=True))
        results = {f'loss': loss.item(), 'mae': mae.item(), 'wape': wape.item()}
        nsml.report(summary=True, scope=locals(), step=n_iter, **results)

        # Save
        if n_iter % 1000 == 0:
            print(f'Save n_iter = {n_iter}')
            nsml.save(n_iter)

    # Load test (Check if load method works well)
    nsml.load(n_iter)
    logging.info('Save & Load test succeed!')


def csr_to_tensor(csr, device):
    coo = csr.tocoo(copy=False)
    values = coo.data
    indices = np.vstack((coo.col, coo.row))  # transpose
    i = torch.tensor(indices, dtype=torch.long, device=device)
    v = torch.tensor(values, dtype=torch.float32, device=device)
    shape = coo.shape[1], coo.shape[0]  # transpose
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


if __name__ == '__main__':
    main()
