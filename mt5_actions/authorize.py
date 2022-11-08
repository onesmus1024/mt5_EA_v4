import MetaTrader5 as mt5

account =160282544
password = "Fxtm1024"
server = "ForexTimeFXTM-Demo01"
path ="C:\\Program Files\\ForexTime (FXTM) MT5\\terminal64.exe"



def login()->bool:
    if not mt5.initialize(path=path,login=account,server=server):
        print("initialize() failed, error code =",mt5.last_error())
    authorized=mt5.login(login=account,password=password,server=server)

    if authorized:
        print(mt5.account_info())
        print("Show account_info()._asdict():")
        account_info_dict = mt5.account_info()._asdict()
        for prop in account_info_dict:
            print("  {}={}".format(prop, account_info_dict[prop]))
        return True
    else:
        print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
        return False