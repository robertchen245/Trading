import vectorbt as vbt
data = vbt.YFData.download('QQQ', start='2016-01-01', end='2026-01-01')
price = data.get("Close")
pf = vbt.Portfolio.from_holding(price, init_cash=100)
print(pf.total_profit())