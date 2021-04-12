# 1-6 스마트스토어 상품 단위 수요 예측
- 스마트스토어의 상품의 일단위 판매량을 예측하는 문제입니다.
- 상품의 예시: https://brand.naver.com/lgcaremall/products/4943642550

## 학습 데이터
- train/train_data/order_hist.npz
  - CSR matrix with the shape (735, 1395407)
  - 스마트스토어의 1,395,407 개의 상품에 대한 735일간 일별(daily) 판매 기록을 학습데이터로 제공하고 있습니다. (모두 사용하지 않으셔도 무방합니다.)
  - 보안 문제로 인해 임의의 난수를 곱하여 제공했습니다.

- train/train_data/price_hist.npz
  - CSR matrix with the shape (735, 1395407)
  - 스마트스토어의 1,395,407 개의 상품에 대한 735일간 일별(daily) 평균 소비자 구매 가격 기록을 학습데이터로 제공하고 있습니다. (모두 사용하지 않으셔도 무방합니다.)
  - 보안 문제로 인해 임의의 난수를 곱하여 제공했습니다.

- train/train_data/prod_feat.csv
  - CSV with the column ['npay_mrc_id', 'prod_catg_1depth_id', 'prod_catg_2depth_id', 'prod_catg_3depth_id', 'prod_catg_4depth_id']
  - 1,395,407 개의 상품에 대한 5가지 피쳐를 제공합니다.
  - 각각 스토어id, 대분류, 중분류, 소분류, 소소분류에 대한 categorical 피쳐입니다.
 
- 데이터 로드 예제 (main.py 참조)
    ```
    USECOLS = ['npay_mrc_id', 'prod_catg_1depth_id', 'prod_catg_2depth_id', 'prod_catg_3depth_id', 'prod_catg_4depth_id']
    train_dataset_path = os.path.join(nsml.DATASET_PATH, 'train', 'train_data')
    order_hists = sp.load_npz(os.path.join(train_dataset_path, 'order_hist.npz'))
    price_hists = sp.load_npz(os.path.join(train_dataset_path, 'price_hist.npz'))
    prod_feat = pd.read_csv(
        os.path.join(train_dataset_path, 'prod_feat.csv'),
        usecols=USECOLS, dtype=int
    ).values
    ```
    
## 테스트 데이터
- 총 12개의 테스트 데이터셋이 존재합니다.
- 각 테스트 데이터셋의 ground truth는 랜덤으로 선택된 500개의 상품에 대한 1주일치 판매 기록입니다.
- 과거 735일간의 기록, 과거 및 테스트기간 동안의 가격 정보, 상품정보를 활용할 수 있습니다.
  - train_order_hist: np.array [500, 735]
  - train_price_hist: np.array [500, 735]
  - test_price_hist: np.array [500, 7]
  - prod_feat: np.array [500, 5]

## 모델
- PyTorch, Keras, Tensorflow, Scikit-learn, Numpy, ... 편하신대로 만드시면 됩니다.
- NSML에 bind되는 infer(train_order_hist, train_price_hist, test_price_hist, prod_feat) 함수의 input 변수를 변경하지 마세요.
- infer() 함수 내부는 변경하셔도 무관합니다.
- infer() 함수는 주어진 input 4개를 활용하여 np.array [500, 7] 형태의 예측치를 return해야 합니다.

## 베이스라인
- (지난 num_weeks 주간의 평균) * level_trend 를 예측으로 사용합니다.
- main.py를 참조해주세요.

## 평가
- 평가는 [WAPE](https://www.baeldung.com/cs/mape-vs-wape-vs-wmape)를 사용합니다
- 12개 테스트 데이터섯 각각에 대해 WAPE를 계산하고, 이에 대한 평균값을 최종 평가로 사용합니다.

## 예상 질문
- Q. 테스트데이터는 학습데이터에 포함되어 있나요?
  - A. 학습데이터의 기간과 테스트데이터의 기간이 다릅니다. 하지만, 테스트데이터의 500개 상품은 학습데이터 1395407개 상품 중 일부입니다.

- Q. 테스트 데이터셋에 사용되는 500개는 모두 같은 상품인가요?
  - A. 아니요. 하지만 일부는 겹칠 수 있습니다.

- Q. 테스트 데이터셋은 어떤 기준으로 500개의 상품을 선택하였나요?
  - A. 1395407개의 상품 중 테스트 기간에 일정 수준의 판매를 보인 상품 위주로 필터링한 후 랜덤으로 선택하였습니다. (지나치게 sparse한 상품은 테스트에서 제외했습니다.)


- Q. 학습 order_hist.npz 데이터가 상당히 sparse합니다. 정상인가요?
  - A. 네. 일부 상품은 특정 며칠에만 팔리기도 합니다. 해당 데이터에 대한 사용 여부는 참가자의 자유입니다.
  
- Q. 학습 데이터에 사용된 상품번호와 테스트 데이터에 사용된 상품번호는 일치하나요?
  - A. 아니요.

- Q. 학습 데이터에 사용된 판매량 및 가격 정보의 scale은 테스트 데이터에서 유지되나요?
  - A. 네.

- Q. 학습 데이터에 사용된 상품 피쳐별 카테고리id는 테스트 데이터에서 유지되나요?
  - A. 네.
