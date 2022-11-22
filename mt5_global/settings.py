import MetaTrader5 as mt5
import datetime
import pytz
Model_type = 'v4_CNN+GRU'
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M1
time_series = 5
Debug = False

timezone = pytz.utc
utc_from = datetime.datetime(2022, 10, 1, tzinfo=timezone)