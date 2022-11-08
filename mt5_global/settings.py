import MetaTrader5 as mt5
import datetime
import pytz
Model_type = 'v4_CNN+GRU'
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
time_series = 15
Debug = False

timezone = pytz.timezone("Etc/UTC")
utc_from = datetime.datetime(2020, 3, 1, tzinfo=timezone)