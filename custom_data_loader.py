import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import test
import psycopg2

def compute_ewm_DD(ret_ser, hl):
    ret_ser_neg = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

def feature_engineer(ret_ser, hls = [20, 60, 120]):
    feat_dict = {}
    for hl in hls:
        DD = compute_ewm_DD(ret_ser, hl)

        # Feature 1: EWM Return
        feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()

        # Feature 2: log(EWM-DD)
        feat_dict[f"DD-log_{hl}"] = np.log(DD)

        # Feature 3: EWM-Sortino Ratio = EWM-Return / EWM-DD
        feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)


    DD_low = compute_ewm_DD(ret_ser, hls[0])
    DD_mid = compute_ewm_DD(ret_ser, hls[1])
    DD_high = compute_ewm_DD(ret_ser, hls[2])

    # Feature 1: EWM-DD(hl = 20)
    feat_dict[f"DD_{hls[0]}"] = DD_low

    # Feature 2: EWM-DD(hl = 20) - EWM-DD(hl = 60)
    feat_dict[f"DD_{hls[0]}_{hls[1]}_diff"] = DD_low - DD_mid

    # Feature 3: EWM-DD(hl = 60) - EWM-DD(hl = 120)
    feat_dict[f"DD_{hls[1]}_{hls[2]}_diff"] = DD_mid - DD_high

    # Feature 4: ReturnEWMA(hl = 120)
    feat_dict[f"ret_{hls[2]}"] = ret_ser.ewm(halflife=hls[2]).mean()

    return pd.DataFrame(feat_dict)

def filter_date_range(df, start_date, end_date):
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df.loc[start_date:]
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df.loc[:end_date]
    return df.copy()

class CustomDataLoader(BaseEstimator):
    def __init__(self, ticker=None, path=None, path_type='excel'or'csv'or'pickle', ticker_col_name = 'Close', hls = [20, 60, 120]):
        self.ticker = ticker
        self.path = path
        self.path_type = path_type
        self.ticker_col_name = ticker_col_name
        self.hls = hls
        if self.ticker and self.path:
            raise ValueError("Only one of ticker or path should be given.")
        if not self.ticker and not self.path:
            raise ValueError("Either ticker or path should be given.")
        
    def load(self, start_date=None, end_date=None):

        if self.ticker:
            connection = psycopg2.connect(
                host = 'XXX',
                port = 0000,
                user = 'XXX',
                password = 'XXX',
                database= 'postgres'
            )

            query = f'''select "Date", "Index", "Close" from public.bmklevels where "Index" in ('{self.ticker}') '''
            df = pd.read_sql(query, connection)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Close'] = df['Close'].astype(float)
            df = df.pivot(index='Date', values='Close', columns='Index')
            # df.rename_axis('date', inplace=True)
            df['ret'] = df[self.ticker].pct_change()
            ret_ser_raw = df[ 'ret']

        elif self.path:
            if self.path_type == 'excel':
                df = pd.read_excel(self-path, parse_dates=['Date'], index_co1='Date')
                df_ret = df[self.ticker_col_name].pct_change().dropna()
                ret_ser_raw = df_ret
            elif self.path_type == 'csv':
                df = pd.read_csv(self.path, parse_dates=['Date'], index_col='Date')
                df_ret = df[self.ticker_col_name].pct_change().dropna()
                ret_ser_raw = df_ret
            elif self.path_type == 'pickle':
                df = pd.read_pickle(self.path)
                df_ret = df[self.ticker_col_name].pct_change().dropna()
                ret_ser_raw = df_ret
            else:
                raise ValueError("Invalid path type. Please choose from 'excel', 'csv', or 'pickle'.")

        start_date = pd.to_datetime(start_date) if start_date else ret_ser_raw.index[0]
        end_date = pd.to_datetime(end_date) if end_date else ret_ser_raw.index[-1]
        ret_ser_raw = ret_ser_raw[start_date:end_date]

        return ret_ser_raw
        
class ExpandingScaler(BaseEstimator):
    def __init__(self, min_periods=30):
        self.min_periods = min_periods

    def fit_transform(self, X):
        return X.expanding(min_periods=self.min_periods).apply(lambda x: (x[-1] -x.mean()) / x.std())
    
class StandardScalerCM(BaseEstimator):
    def init_scaler(self) :
        return StandardScaler()

    def fit(self, X):
        self.scaler = self.init_scaler().fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
class ExpandingScaler(BaseEstimator):

    def __init__(self, min_periods=30, mul=3):
        self.min_periods = min_periods
        self.mul = mul

    def fit_transform(self, X):
        X_mean = X.expanding(min_periods=self.min_periods).mean()
        X_std = X.expanding(min_periods=self.min_periods).std()
        lb = X_mean - X * X_std()*self.mul
        ub = X_mean + X * X_std()*self.mul
        return np.where(X<lb, lb, np.where(X>ub, ub, X))
    

        
            



    
        