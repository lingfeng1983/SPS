"""测试新浪 ETF 专用接口 fund_etf_hist_sina 作为 ETF 备用源。"""
import sys
sys.path.insert(0, r"D:\SPS")
import akshare as ak

df = ak.fund_etf_hist_sina(symbol="sh510300")
print("OK rows:", len(df), "cols:", df.columns.tolist())
print(df.tail(2).to_string())
