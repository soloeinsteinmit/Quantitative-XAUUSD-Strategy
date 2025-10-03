import os
from dotenv import load_dotenv

load_dotenv()

DEMO_ACCOUNT_NUMBER = os.getenv("DEMO_ACCOUNT_NUMBER")
PASSWORD=os.getenv("PASSWORD")
SERVER=os.getenv("SERVER")

HYP_A_XGB_ACCOUNT_INFO:dict = {
    "lot_size": float(os.getenv("HYP_A_XGB_ACCOUNT_LOT_SIZE")),
    "mt5_account": int(os.getenv("HYP_A_XGB_ACCOUNT_NUMBER")),
    "mt5_account_name": os.getenv("HYP_A_XGB_ACCOUNT_NAME"),
    "mt5_account_type": os.getenv("HYP_A_XGB_ACCOUNT_TYPE"),
    "mt5_password": os.getenv("HYP_A_XGB_ACCOUNT_PASSWORD"),
    "mt5_server": os.getenv("HYP_A_XGB_ACCOUNT_SERVER"),
    "magic_number": int(os.getenv("HYP_A_XGB_ACCOUNT_MAGIC_NUMBER")),
}
