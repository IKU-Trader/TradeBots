
import math
import os
import ccxt
#from crypto_data_fetcher import GmoFetcher
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Meiryo')

#import numba
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import seaborn as sns
import talib

from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit



def calc_features(df):
    open = df['op'].values
    high = df['hi'].values
    low = df['lo'].values
    close = df['cl'].values
    volume = df['volume'].values
    orig_columns = df.columns

    hilo = (high + low) / 2
    up, mid, lower = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] = up - hilo
    df['BBANDS_middleband'] = mid - hilo
    df['BBANDS_lowerband'] = lower - hilo
    df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
    df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    df['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
    df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df['AROON_down'], df['AROON_up'] = talib.AROON(high, low, timeperiod=14)
    df['AROON_osc'] = talib.AROONOSC(high, low, timeperiod=14)
    df['BOP'] = talib.BOP(open, high, low, close)
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['DX'] = talib.DX(high, low, close, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # skip MACDEXT MACDFIX たぶん同じなので
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    df['STOCH1_slowk'], df['STOCH2_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCH3_fastk'], df['STOCH4_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = talib.TRIX(close, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    df['AD'] = talib.AD(high, low, close, volume)
    df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(close, volume)

    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(high, low, close)

    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    df['BETA'] = talib.BETA(high, low, timeperiod=5)
    df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)

    return df
    
features = sorted([
    'ADX',
    'ADXR',
    'APO',
    'AROON_down',
    'AROON_up',
    'AROON_osc',
    'CCI',
    'DX',
    'MACD_macd',
    'MACD_macdsignal',
    'MACD_macdhist',
    'MFI',
#     'MINUS_DI',
#     'MINUS_DM',
    'MOM',
#     'PLUS_DI',
#     'PLUS_DM',
    'RSI',
    'STOCH1_slowk',
    'STOCH2_slowd',
    'STOCH3_fastk',
#     'STOCHRSI_fastd',
    'ULTOSC',
    'WILLR',
#     'ADOSC',
#     'NATR',
    'HT_DCPERIOD',
    'HT_DCPHASE',
    'HT_PHASOR_inphase',
    'HT_PHASOR_quadrature',
    'HT_TRENDMODE',
    'BETA',
    'LINEARREG',
    'LINEARREG_ANGLE',
    'LINEARREG_INTERCEPT',
    'LINEARREG_SLOPE',
    'STDDEV',
    'BBANDS_upperband',
    'BBANDS_middleband',
    'BBANDS_lowerband',
    'DEMA',
    'EMA',
    'HT_TRENDLINE',
    'KAMA',
    'MA',
    'MIDPOINT',
    'T3',
    'TEMA',
    'TRIMA',
    'WMA',
])
  
def fillZeroIfNan(df, columns):
    for column in columns:
        df2 = df[column]
        for i in range(len(df2)):
            if np.isnan(df2.iat[i]):
                df2.iat[i] = 0  

def calc_force_entry_price(entry_price=None, lo=None, pips=None):
    y = entry_price.copy()
    y[:] = np.nan
    force_entry_time = entry_price.copy()
    force_entry_time[:] = np.nan
    for i in range(entry_price.size):
        for j in range(i + 1, entry_price.size):
            if round(lo[j] / pips) < round(entry_price[j - 1] / pips):
                y[i] = entry_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


def trade(df, name):

    # 呼び値 (取引所、取引ペアごとに異なるので、適切に設定してください)
    pips = 1

    # ATRで指値距離を計算します
    limit_price_dist = df['ATR'] * 0.5
    limit_price_dist = np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips

    # 終値から両側にlimit_price_distだけ離れたところに、買い指値と売り指値を出します
    df['buy_price'] = df['cl'] - limit_price_dist
    df['sell_price'] = df['cl'] + limit_price_dist

    # Force Entry Priceの計算
    df['buy_fep'], df['buy_fet'] = calc_force_entry_price(
        entry_price=df['buy_price'].values,
        lo=df['lo'].values,
        pips=pips,
        )

    # calc_force_entry_priceは入力と出力をマイナスにすれば売りに使えます
    df['sell_fep'], df['sell_fet'] = calc_force_entry_price(
        entry_price=-df['sell_price'].values,
        lo=-df['hi'].values, # 売りのときは高値
        pips=pips,
        )
    df['sell_fep'] *= -1

    horizon = 1 # エントリーしてからエグジットを始めるまでの待ち時間 (1以上である必要がある)
    fee = df['fee'] # maker手数料

    # 指値が約定したかどうか (0, 1)
    df['buy_signal'] = ((df['buy_price'] / pips).round() > (df['lo'].shift(-1) / pips).round()).astype('float64')
    df['sell_signal'] = ((df['sell_price'] / pips).round() < (df['hi'].shift(-1) / pips).round()).astype('float64')

    # yを計算
    df ['long_entry'] = np.where(
                            df['buy_signal'],
                            1,
                            0
                            )
    df ['short_entry'] = np.where(
                            df['sell_signal'],
                            1,
                            0
                            )
    
    
    df['long_ror'] = np.where(
                            df['buy_signal'],
                            df['sell_fep'].shift(-horizon) / df['buy_price'] - 1, #- 2 * fee,
                            0
                            )
    
    df['long_open'] = np.where(
                            df['buy_signal'],
                            df['buy_price'],
                            0
                            )
    
    df['long_close'] = np.where(
                            df['buy_signal'],
                            df['sell_fep'].shift(-horizon),
                            0
                            )
    
    df['short_ror'] = np.where(
                            df['sell_signal'],
                            -(df['buy_fep'].shift(-horizon) / df['sell_price'] - 1), #,- 2 * fee,
                            0
                            )
    
    df['short_open'] = np.where(
                            df['sell_signal'],
                            df['sell_price'],
                            0
                            )
    
    df['short_close'] = np.where(
                            df['sell_signal'],
                            df['buy_fep'].shift(-horizon),
                            0
                            )
    
    
    fillZeroIfNan(df, ['buy_signal', 'sell_signal', 'long_entry', 'short_entry', 'long_open', 'short_open', 'long_close',  'short_close', 'long_ror', 'short_ror'])

    # バックテストで利用する取引コストを計算
    df['buy_cost'] = np.where(
                            df['buy_signal'],
                            df['buy_price'] / df['cl'] - 1 + fee,
                            0
                            )
    df['sell_cost'] = np.where(
                            df['sell_signal'],
                            -(df['sell_price'] / df['cl'] - 1) + fee,
                            0
                            )

    print('約定確率を可視化。時期によって約定確率が大きく変わると良くない。')
    df['buy_signal'].rolling(1000).mean().plot(label='買い')
    df['sell_signal'].rolling(1000).mean().plot(label='売り')
    plt.title('約定確率の推移')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

    print('エグジットまでの時間分布を可視化。長すぎるとロングしているだけとかショートしているだけになるので良くない。')
    df['buy_fet'].rolling(1000).mean().plot(label='買い')
    df['sell_fet'].rolling(1000).mean().plot(label='売り')
    plt.title('エグジットまでの平均時間推移')
    plt.legend(bbox_to_anchor=(1.2, 1))
    plt.show()

    df['buy_fet'].hist(alpha=0.3, label='買い')
    df['sell_fet'].hist(alpha=0.3, label='売り')
    plt.title('エグジットまでの時間分布')
    plt.legend(bbox_to_anchor=(1.2, 1))
    plt.show()

    print('毎時刻、この執行方法でトレードした場合の累積リターン')
    df['long_ror'].cumsum().plot(label='買い')
    df['short_ror'].cumsum().plot(label='売り')
    plt.title('累積リターン')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

    #df.to_pickle('df_y.pkl')    
    columns = ['timestamp', 'op', 'hi', 'cl', 'fee', 'ATR', 'buy_price', 'sell_price', 'buy_signal', 'sell_signal', 'long_entry', 'long_open', 'long_close', 'long_ror', 'short_entry', 'short_open', 'short_close', 'short_ror']
    df2 = df[columns]
    df2.to_csv('./report/richbtc_' + name  + '.csv', index=False)

def ml(df, name):
    columns = ['timestamp', 'op', 'hi', 'cl', 'fee'] + features + ['buy_price', 'sell_price', 'buy_signal', 'sell_signal', 'long_entry', 'long_open', 'long_close', 'long_ror', 'short_entry', 'short_open', 'short_close', 'short_ror']
    df = df[columns]
    df.to_csv('./report/richbtc_before_drop.csv', index=False)
    df = df.dropna()
    df.to_csv('./report/richbtc_after_drop.csv', index=False)
    

    # モデル (コメントアウトで他モデルも試してみてください)
    # model = RidgeCV(alphas=np.logspace(-7, 7, num=20))
    model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)

    # アンサンブル (コメントアウトを外して性能を比較してみてください)
    # model = BaggingRegressor(model, random_state=1, n_jobs=1)

    # 本番用モデルの学習 (このチュートリアルでは使わない)
    # 実稼働する用のモデルはデータ全体で学習させると良い
    #model.fit(df[features], df['y_buy'])
    #joblib.dump(model, 'model_y_buy.xz', compress=True) 
    #model.fit(df[features], df['y_sell'])
    #joblib.dump(model, 'model_y_sell.xz', compress=True)

    # 通常のCV
    cv_indicies = list(KFold().split(df))
    # ウォークフォワード法
    # cv_indicies = list(TimeSeriesSplit().split(df))
    
    #df['long_ror_pred'] = my_cross_val_predict(model, df[features].values, df['long_ror'].values, cv=cv_indicies)
    #df['short_ror_pred'] = my_cross_val_predict(model, df[features].values, df['short_ror'].values, cv=cv_indicies)

    df['long_ror_pred'] = predict(model, df[features].values, df['long_ror'].values, 0.5)
    df['short_ror_pred'] = predict(model, df[features].values, df['short_ror'].values, 0.5, debug=True)
    # 予測値が無い(nan)行をドロップ
    #df = df.dropna()
    
    print('毎時刻、y_predがプラスのときだけトレードした場合の累積リターン')
    df[df['long_ror_pred'] > 0]['long_ror'].cumsum().plot(label='買い')
    df[df['short_ror_pred'] > 0]['short_ror'].cumsum().plot(label='売り')
    #(df['y_buy'] * (df['y_pred_buy'] > 0) + df['y_sell'] * (df['y_pred_sell'] > 0)).cumsum().plot(label='買い+売り')
    plt.title('累積リターン')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()


    columns = ['timestamp', 'op', 'hi', 'cl', 'fee', 'buy_price', 'sell_price', 'buy_signal', 'sell_signal', 'long_entry', 'long_open', 'long_close', 'long_ror', 'short_entry', 'short_open', 'short_close', 'short_ror', 'long_ror_pred', 'short_ror_pred']
    df2 = df[columns + features]
    df2.to_csv('./report/ml_richbtc_' + name + '.csv', index=False)
    #df.to_pickle('df_fit.pkl') 


def saveArray(array, filepath):
    rows = array.shape[0]
    if len(array.shape) > 1:
        cols = array.shape[1]
    else:
        cols = 1
    f = open(filepath, mode='w')
    for row in range(rows):
        s = ''
        for col in range(cols):
            if cols == 1:
                s += str(array[row]) + ','
            else:
                s += str(array[row, col]) + ','
        f.write(s + '\n')
    f.close()
        
def predict(estimator, X, y, rate, debug=False):
    n = len(X)
    m = int (float(n) * rate)
    trainX = X[0:m]
    trainY = y[0:m]
    predY = np.full(n, np.nan)
    estimator.fit(trainX, trainY)
    testX = X[m:]
    testY = y[m:]
    predY[m:] = estimator.predict(testX)
    
    if debug:
        saveArray(trainX, './report/richbtc_trainX.csv')
        saveArray(trainY, './report/richbtc_trainY.csv')
        saveArray(testX, './report/richbtc_testX.csv')
        saveArray(predY, './report/richbtc_pred.csv')
    return predY

# OOS予測値を計算
def my_cross_val_predict(estimator, X, y=None, cv=None):
    y_pred = y.copy()
    y_pred[:] = np.nan
    for train_idx, val_idx in cv:
        estimator.fit(X[train_idx], y[train_idx])
        y_pred[val_idx] = estimator.predict(X[val_idx])
    return y_pred
   
    
    
def readCsv(filepath):
    f = open(filepath)
    l = f.readline()
    l = l.strip()
    columns = l.split(',')
    n = len(columns)
    data = []
    while l:
        values = l.split(',')
        if len(values) != n:
            l = f.readline()
            continue
        d = []
        error = False
        for i in range(n):
            if i == 0:
                d.append(values[0])
            else:
                try:
                    num = float(values[i])
                    d.append(num)
                except:
                    error = True
                    break
        if not error:
            data.append(d)
        l = f.readline()
        
    f.close()
    df = pd.DataFrame(data=data, columns=columns)
    return df
    

if __name__ == '__main__':
    path = "./data/bitflyer/btcjpy_m15.csv"
    df = readCsv(path)
    #df = df.dropna()
    dirpath, filepath = os.path.split(path)
    name, _ = os.path.splitext(filepath)
    df = calc_features(df)
    trade(df, name)
    ml(df, name)
    