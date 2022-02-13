
# 使い方
# 指定の期間だけほしい時。
# python bf_get_ohlcv.py 2019-05-01 2020-01-01

# 未確定足を含む最新の足までほしい時は、最初の時刻だけ指定。
# python bf_get_ohlcv.py 2019-05-01

# 引数の時間の文字列(例：2019-05-01)は、ろくにエラー判定していないので、きちんと間違えずに指定してください。

import requests
import sys
from logging import getLogger,INFO,DEBUG,StreamHandler,FileHandler,Formatter
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import time
import os
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse
import csv

DELETE_MENTE_TIME_DATA = True       # Trueにしておくとメンテ時間のデーターは削除する。
OHLCV_FILE_DIR = 'ohlcv_spot_files/'      # 保存フォルダ。最後にスラッシュ

UTC = timezone.utc
JST = timezone(timedelta(hours=+9), 'JST')

# -----------------------------------------------------------------------------
def epoch_to_dt(epoch, tz=UTC):
    return datetime.fromtimestamp(epoch, tz)

# -----------------------------------------------------------------------------
def str_to_dt(string:str, day_start:bool=True):
    """
    2018-03-17のような JST 時刻文字列をdatetimeに変換
    :param string: %Y-%m-%d
    :param day_start trueだと、00:00:00 にし、falseだと23:59:59にする
    :return: datetime
    """
    # yy = int(string[0:4])
    # mm = int(string[5:7])
    # dd = int(string[8:10])
    if day_start:
        dt = parse(string)
        # return datetime(year=yy, month=mm, day=dd, hour=0, minute=0, second=0, microsecond=0, tzinfo=JST)
    else:
        dt = parse(string + " 23:59:00")
        # return datetime(year=yy, month=mm, day=dd, hour=23, minute=59, second=0, microsecond=0, tzinfo=JST)
    dt = dt.replace(tzinfo=JST)
    # dt.astimezone(JST)
    return dt

# -----------------------------------------------------------------------------
def setup_logger(level:str='INFO'):
    """
    logger作成
    :param level: loggerのレベル。INFO or DEBUG
    :return: logger
    """

    logger = getLogger(__name__)
    handler = StreamHandler()
    if level.upper() == 'INFO':
        log_level = INFO
    elif level.upper() == 'DEBUG':
        log_level = DEBUG
    else:
        print('logging_levell 指定エラー')
        log_level = INFO
    handler.setLevel(log_level)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger

# -----------------------------------------------------------------------------
def logger_add_filehandler(logger, log_folder:str):
    """
    loggerの出力ファイル指定。
    :param logger:
    :param log_folder:出力フォルダ
    :return:
    """

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    fh = FileHandler(log_folder + 'backtest_' + time.strftime('%Y-%m-%d') + '.log')
    fh.setFormatter(Formatter(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)  # 日ごとのファイルハンドラを追加

    return logger

# -----------------------------------------------------------------------------
def _query(query_url, headers):
    response = requests.get(query_url, headers=headers)
    return response.json()

# -----------------------------------------------------------------------------
def bf_get_ohlcv(before_time:int=0):
    """
    before_epocより前のohlcvを取得。降順。
    :param before_time: UNIX time x1000。0だと最新の未確定足より取得。この場合は２本分しかくれない。
    :return:
    """

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
        if before_time != 0:
            res = _query('https://lightchart.bitflyer.com/api/ohlc?symbol=BTC_JPY&before=' + str(before_time)+ '&period=m', headers)
        else:
            res = _query('https://lightchart.bitflyer.com/api/ohlc?symbol=BTC_JPY&period=m', headers)

        # [0] : time
        # [1] : open
        # [2] : high
        # [3] : low
        # [4] : close
        # [5] : total_volume
        # [6] : asks board total vol
        # [7] : bids board total vol
        # [8] : asks executions vol
        # [9] : bids executions vol
        if type(res) == list and len(res) > 0 and len(res[0]) == 10:
            return res
        else:
            logger.error('bf_get_ohlcv() error!')
            logger.error(res)
            return None

    except Exception as e:
        logger.error(e)
        return None

# -----------------------------------------------------------------------------
def get_ohlcv(start_jst_dt:datetime, end_jst_dt:datetime):
    """
    start_dt～end_dayまでのOHLCVを月単位で保存する。
    :param start_dt:
    :param end_dt:
    :return:
    """
    # start_utc_dt = start_jst_dt.astimezone(dateutils.UTC)
    # end_utc_dt = end_jst_dt.astimezone(dateutils.UTC)


    ohlcv_list = []

    # bfからのレスポンスは任意の時間で取得できず、９時、21時、の区切りとなっているので、最初の取得は１２時間前倒して取得して、いらない部分を捨てる。
    if end_jst_dt !=0:
        end_epoc = int(end_jst_dt.timestamp()) * 1000
        kari_end_dt = end_jst_dt + timedelta(hours=12)
        kari_end_epoc = int(kari_end_dt.timestamp())
        before_time = kari_end_epoc*1000
        find = False
        while True:
            ret = bf_get_ohlcv(before_time)
            for i in range(len(ret)):
                if ret[i][0] <= end_epoc:
                    ohlcv_list.extend(ret[i:])
                    find = True
                    break
            if find:
                break
            before_time = ret[-1][0]-60000      # 最後の取得時間から１分時間を減らして取得
            time.sleep(0.2)

        before_time = ret[-1][0]-60000
        now_month = end_jst_dt.month
        now_year = end_jst_dt.year
    else:
        # 最新の未確定足から読み込む
        before_time = 0
        now_month = datetime.now().month
        now_year = datetime.now().year

    # bfの取得は、新しい日付から古い日付順に帰ってくる。
    while True:
        ret = bf_get_ohlcv(before_time)
        
        if ret == None:
            print('不正なレスポンスが返ってきましたので終了します。')
            sys.exit(1)
            
        if before_time == 0 and len(ret) ==2:
            # 最新の未確定足を捨てる。
            ret.pop(0)

        # ret_start_time = int(str(ret[0][0])[:10]) # タイムスタンプのミリ秒を削除
        # ret_start_dt = epoch_to_dt(ret_start_time, JST)

        ret_end_time = int(str(ret[-1][0])[:10]) # タイムスタンプのミリ秒を削除
        ret_end_dt = epoch_to_dt(ret_end_time, JST)

        if DELETE_MENTE_TIME_DATA:
            # 前処理で、openeがNoneのものは、メンテなので削除しておく
            # リストの後ろから回して削除していく。
            for i in reversed(range(len(ret))):     # range(len(ret)-1, -1, -1)と同じ。
                if ret[i][1] == None:
                    del ret[i]


        # 最後の月
        if ret_end_dt <= start_jst_dt:
            # さかのぼって、何個目からstart_jst_dtになっているか調べる
            for i in range(len(ret) - 2, -1, -1):
                timestump = int(str(ret[i][0])[:10])  # タイムスタンプのミリ秒を削除
                timestump_dt = epoch_to_dt(timestump, JST)
                if timestump_dt >= start_jst_dt:
                    ohlcv_list.extend(ret[0:i+1])
                    break

            new_ohlcv_list = [ [str(datetime.fromtimestamp(ohlcv[0]/1000)), ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5], ohlcv[9], ohlcv[8] ] for ohlcv in ohlcv_list[::-1]]
            # ファイル書き込み
            filename = 'ohlcv_1min_{:02}_{:02}.csv'.format(now_year, now_month)
            with open(OHLCV_FILE_DIR+filename, 'w', newline="") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(["date","open","high","low","close","volume","buy_vol","sell_vol"])
                writer.writerows(new_ohlcv_list)
            logger.info(filename+' 書き込み完了')
            return

        # 月がかわったのでファイル保存
        if ret_end_dt.month != now_month:
            # さかのぼって、何個目から月が変わったかしらべる
            for i in range(len(ret)-2,-1,-1):
                timestump = int(str(ret[i][0])[:10])  # タイムスタンプのミリ秒を削除
                # ret_end_time -= 32400  # 日本時間に変更
                timestump_dt = epoch_to_dt(timestump, JST)
                if timestump_dt.month == now_month:
                    ohlcv_list.extend(ret[0:i+1])
                    break

            new_ohlcv_list = [ [str(datetime.fromtimestamp(ohlcv[0]/1000)), ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], ohlcv[5], ohlcv[9], ohlcv[8] ] for ohlcv in ohlcv_list[::-1]]

            # ファイル書き込み
            filename = 'ohlcv_1min_{:02}_{:02}.csv'.format(now_year, now_month)
            with open(OHLCV_FILE_DIR+filename, 'w', newline="") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(["date","open","high","low","close","volume","buy_vol","sell_vol"])
                writer.writerows(new_ohlcv_list)
            logger.info(filename+' 書き込み完了')

            # 途中で月が変わったので、その部分のlistは保持して次の月の受信へ
            ohlcv_list = ret[i+1:]
            now_month = ret_end_dt.month
            now_year = ret_end_dt.year

        else:
            ohlcv_list.extend(ret)

        before_time = ret[-1][0]-60000      # 最後の取得時間から１分時間を減らして取得
        time.sleep(0.2)

# -----------------------------------------------------------------------------
def main():

    # コマンドライン引数の処理。第一引数；開始日付、第2引数；終了日付（０だと最新まで）
    args = sys.argv
    if len(args) == 1:
        start_dt = datetime.now()
        start_dt = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_dt = start_dt.astimezone(JST)
        print('{}/{}/{} から、最新の足まで取得します。'.format(start_dt.year, start_dt.month, start_dt.day))
        end_dt = 0      # 最新の未確定足から読み込む
    elif len(args) == 2:
        start_dt = str_to_dt(str(args[1]), True)
        # end_dt = dateutils.epoch_to_dt(time.time())
        end_dt = 0      # 最新の未確定足から読み込む
    elif len(args) == 3:
        start_dt = str_to_dt(str(args[1]), True)
        end_dt = str_to_dt(str(args[2]), False)
    else:
        print('引数がちがいます。')
        sys.exit(1)

    get_ohlcv(start_dt, end_dt)

    return


# -------------------------------------------------------------------------
if __name__ == '__main__':

    # loggerの準備
    logger = setup_logger('INFO')
    # logger = logger_add_filehandler(logger, './')

    # 保存フォルダを作成
    if not os.path.exists('./'+OHLCV_FILE_DIR):
        os.makedirs(os.path.dirname('./'+OHLCV_FILE_DIR))

    main()
