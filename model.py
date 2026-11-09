# 1. 코딩 준비
!pip install -U finance-datareader
import numpy as np
import FinanceDataReader as fdr
from scipy.optimize import minimize
from datetime import datetime

# 2. 데이터 준비
df = fdr.DataReader('KS11', '2021-03-15')
e_rate = 100 * np.log(df['Close'] / df['Close'].shift(1)).dropna()

# 3. 모델 정의 및 로그우도함수
def mrs_v(parameters, e_rate):
    v0, v1, p00, p11 = parameters
    n = len(e_rate)
    p0 = np.zeros(n)
    log_lik_sum = 0
    p0[0] = 0.5
    for t in range(1, n):
        p0_pred = p0[t-1] * p00 + (1 - p0[t-1]) * (1 - p11)
        f0 = (1 / np.sqrt(2 * np.pi * v0)) * np.exp(-0.5 * e_rate.iloc[t]**2 / v0)
        f1 = (1 / np.sqrt(2 * np.pi * v1)) * np.exp(-0.5 * e_rate.iloc[t]**2 / v1)
        lik = p0_pred * f0 + (1 - p0_pred) * f1
        log_lik_sum += np.log(max(lik, 1e-10))
        p0[t] = (p0_pred * f0) / lik
    return -log_lik_sum

# 4. 최적화 및 상세 결과 도출
init_guess = [0.5, 3.0, 0.95, 0.90]
bnds = [(0.001, 2), (2, 144), (0.01, 0.95), (0.01, 0.95)]
res = minimize(mrs_v, init_guess, args=(e_rate,), bounds=bnds)

if res.success:
    v0_opt, v1_opt, p00_opt, p11_opt = res.x
    def get_last_state(parameters, e_rate):
        v0, v1, p00, p11 = parameters
        p0 = 0.5
        for r in e_rate:
            p0_prior = p0 * p00 + (1 - p0) * (1 - p11)
            f0 = (1 / np.sqrt(2 * np.pi * v0)) * np.exp(-0.5 * r**2 / v0)
            f1 = (1 / np.sqrt(2 * np.pi * v1)) * np.exp(-0.5 * r**2 / v1)
            p0 = (p0_prior * f0) / (p0_prior * f0 + (1 - p0_prior) * f1)
        return p0

    last_p0 = get_last_state(res.x, e_rate)
    next_p0 = last_p0 * p00_opt + (1 - last_p0) * (1 - p11_opt)
    next_p1 = 1 - next_p0
    next_v = np.sqrt(next_p0 * v0_opt + next_p1 * v1_opt)

    print("=" * 45)
    print(f" 분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 45)
    print(f" 1. 저변동성 국면 변동성(σ0): {np.sqrt(v0_opt):.4f}")
    print(f" 2. 고변동성 국면 변동성(σ1): {np.sqrt(v1_opt):.4f}")
    print("-" * 45)
    print(f" [내일 국면 예측 확률]")
    print(f" - 저변동성 유지 확률: {next_p0:.2%}")
    print(f" - 고변동성 진입 확률: {next_p1:.2%}")
    print("-" * 45)
    print(f" >> 내일 KOSPI 최종 예상 변동성: {next_v:.4f}")
    print("=" * 45)

# 5-1. LGBM 모델 학습 및 예측
!pip install -U optuna lightgbm   # 외부 라이브러리를 인터넷에서 다운로드하여 설치
import optuna  # 직전에 다운로드한 도구에서 optuna를 불러옴
import lightgbm as lgb # LGBM 라이브러리를 불러옴
from lightgbm import LGBMRegressor  # LightGBM에서 회귀모델을 꺼내옴
from sklearn.metrics import mean_squared_error  # MSE 계산함수 불러오기
import pandas as pd  # 판다스를 pd라 함


# 5-2. LGBM 학습을 위한 전체 기간의 p0(국면 확률) 데이터 생성
n = len(e_rate)  # 전체 수익률 데이터 개수
p0_series = np.zeros(n)  # 0이 n개 들어있는 리스트 생성
p0_series[0] = 0.5  # 첫째날 확률을 0.5로 설정
for t in range(1, n):   # 둘째날부터 마지막날까지 국면 0일 확률 계산
    p_prior = p0_series[t-1] * p00_opt + (1 - p0_series[t-1]) * (1 - p11_opt)
    f0 = (1 / np.sqrt(2 * np.pi * v0_opt)) * np.exp(-0.5 * e_rate.iloc[t]**2 / v0_opt)
    f1 = (1 / np.sqrt(2 * np.pi * v1_opt)) * np.exp(-0.5 * e_rate.iloc[t]**2 / v1_opt)
    p0_series[t] = (p_prior * f0) / (p_prior * f0 + (1 - p_prior) * f1)  # MRS에서 구한 내일의 변동성을 매일로 변환


# 5-3. LGBM이 학습할 수 있도록 데이터를 한곳에 표로 정리
df_lgbm = pd.DataFrame(index=e_rate.index)
df_lgbm['e_rate'] = e_rate
df_lgbm['mrs_p0'] = p0_series
df_lgbm['vol_lag1'] = e_rate.abs().shift(1)
df_lgbm['target'] = e_rate.abs().shift(-1).rolling(5).mean()
df_lgbm.dropna(inplace = True)


# 5-4. 데이터 분할
x = df_lgbm[['e_rate', 'mrs_p0', 'vol_lag1']]
y = df_lgbm['target']
split = int(len(x) * 0.8)
x_train, x_test = x.iloc[:split], x.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# 6-1. Optuna를 이용한 하이퍼파라미터 최적화
def optuna_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 2, 64),
        'lambda_1': trial.suggest_float('lambda_1', 1e-8, 10.0, log=True),
        'lambda_2': trial.suggest_float('lambda_2', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0, 5.0),
    }


    model = LGBMRegressor(**params)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(np.mean((y_test - preds)**2))
    return rmse

# 6-2. 최적의 파라미터 찾기 실행
print('optuna를 이용한 최적의 파라미터 탐색중...')
study = optuna.create_study(direction='minimize')
study.optimize(optuna_objective, n_trials=30)

# 6-3. Optuna가 찾은 최적의 파라미터를 lgbm_params에 저장
lgbm_params = study.best_params  # study 객체에서 가장 낮은 RMSE를 기록했던 최적의 하이퍼파라미터 조합을 가져와 변수에 저장
lgbm_params['objective'] = 'regression'  #모델의 목적을 회귀로 설정
lgbm_params['metric'] = 'rmse'  # 성능 평가 기준을 RMSE로 설정
lgbm_params['verbosity'] = -1  # 출력되는 메세지를 최소화하여 결과만 출력함

# 6-4. LGBM 전용 규격의 데이터로 변환
train_ds = lgb.Dataset(x_train, label=y_train)  # 데이터를 빠르고 정확하게 계산하도록 최적화한 데이터규격
test_ds = lgb.Dataset(x_test, label=y_test, reference=train_ds) # test용 데이터를 lgbm 전용 포맷으로 변환

# 7. 최종 LGBM 모델 정의
lgbm_model = lgb.train(lgbm_params, train_ds, num_boost_round=1000, # optuna로 정한 파라미터, 학습용 데이터 입력, 최대 1000번 반복
                       valid_sets=[train_ds, test_ds],  # 학습용 오차와 테스트 오차를 동시에 점검
                       callbacks=[lgb.early_stopping(stopping_rounds=30)])  # test의 오차가 30번 연속으로 좋아지지 않으면 학습을 종료

 # 8-1. 최종 결과 도출
lgbm_pred = lgbm_model.predict(x.iloc[[-1]])[0]  # 오늘까지의 데이터를 입력값으로 하여 내일의 변동성 수치 도출

# 8-2. 보정 오차 계산 (LGBM 예측치 - MRS 예측치)
correction_value = lgbm_pred - next_v  # lgbm이 mrs 결과를 얼마나 수정했는지를 보여줌

# 8-3. 분석 기준일 설정 (현재 시간)
now = datetime.now().strftime('%Y-%m-%d')


print("\n" + "=" * 45)
print(f" [최종 하이브리드 변동성 분석 결과]")
print(f" ※ 분석 기준일: {datetime.now().strftime('%Y-%m-%d')}")
print("-" * 45)
print(f" 1. MRS 모델 기초 예측치 (A) : {next_v:.4f}")
print(f" 2. LGBM 모델 보정 예측치  : {correction_value:+.4f}")
print("-" * 45)
print(f" >> 내일 KOSPI 최종 확정 변동성: {lgbm_pred:.4f}")
print("=" * 45)
