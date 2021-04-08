
import os
import argparse
import pickle as pkl
import logging

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import nsml

FORECAST_LEN = 7
USECOLS = [
    'npay_mrc_id', 'prod_catg_1depth_id', 'prod_catg_2depth_id',
    'prod_catg_3depth_id', 'prod_catg_4depth_id'
]


def bind_model(model, device):
    def save(dir_name):
        params_fname = os.path.join(dir_name, 'params.pkl')
        with open(params_fname, 'wb') as f:
            pkl.dump(model.get_params(), f)

    def load(dir_name):
        params_fname = os.path.join(dir_name, 'params.pkl')
        with open(params_fname, 'rb') as f:
            params = pkl.load(f)
        model.load_params(params)

    def infer(train_order_hist, train_price_hist, test_price_hist, prod_feat):
        """
        :params
            train_order_hist (T_H, N): The order quantity of N=500 products for the last T_H=735 days.
            train_price_hist (T_H, N): The price of N=500 products for the last T_H=735 days.
            test_price_hist (T_F, N): The price of N=500 products for the next T_F=7 days.
            prod_feat (N, F): The F=5 categorical features of each product.
        """
        return model(train_order_hist, train_price_hist, test_price_hist, prod_feat)

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


class SimpleBaseline:
    def __init__(self, num_weeks=1, level_trend=1.0):
        self.params = {
            'num_weeks': num_weeks,
            'level_trend': level_trend
        }

    def train(self, order_hists, price_hists, prod_feat):
        pass

    def get_params(self):
        return self.params

    def load_params(self, params):
        self.params = params

    def __call__(self, train_order_hist, train_price_hist, test_price_hist, prod_feat):
        nw = self.params['num_weeks']
        lt = self.params['level_trend']
        order_hist = train_order_hist[-FORECAST_LEN*nw:].reshape(nw, FORECAST_LEN, -1)
        order_pred = np.mean(order_hist, axis=0) * lt
        return order_pred


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
    device = torch.device(
        'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    )
    logging.getLogger().setLevel(logging.INFO)

    # Bind model
    model = SimpleBaseline()
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

    logging.info('Loading data is finished!')

    # Make model
    num_weeks = 3
    level_trend = 1.1
    model = SimpleBaseline(num_weeks, level_trend)
    model.train(order_hists, price_hists, prod_feat)  # fake train

    # Save and load test
    nsml.save('null')
    nsml.load('null')
    logging.info('Save & Load test succeed!')


if __name__ == '__main__':
    main()
