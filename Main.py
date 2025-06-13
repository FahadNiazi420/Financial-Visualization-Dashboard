import matplotlib.pyplot as plt
from MC import MonteCarloInputSimulator
from FCFF import FCFFModel

fcff = FCFFModel( stock = 'NVDA')
print(fcff.bloomberg_data)

if fcff.stock == 'NVDA':
    user_input = {
    "revenue_growth": [0.10, 0.10, 0.10, 0.10, 0.10, 0.0986, 0.0972, 0.0858, 0.0644, 0.043,0.043],
    "operating_margin": [0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45],
    "tax_rate": [ 0.142,0.142,0.142,0.142,0.142,0.142,0.142,0.142,0.142,0.142,0.142],  # Hier noch eins raus und effective tax rate nutzen
    "reinvestment_rate": [2,2,2,2,2,2,2,2,2,2],
    "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840, 0.0840],
     "roic_tv": 0.2 }
elif fcff.stock == 'MSFT':
    user_input = {
    "revenue_growth": [0.15, 0.15, 0.15, 0.15, 0.15, 0.1286, 0.1072, 0.0858, 0.0644, 0.043,0.043],
    "operating_margin": [0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45],
    "tax_rate": [ 0.182,0.182,0.182,0.182,0.182,0.182,0.182,0.182,0.182,0.182,0.182],  # Hier noch eins raus und effective tax rate nutzen
    "reinvestment_rate": [2,2,2,2,2,2,2,2,2,2],
    "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840, 0.0840],
     "roic_tv": 0.2 }

x, y = fcff.build_forecast_df(user_inputs=user_input)
print('Output y: \n')
print(y)
z = fcff.calculate_valuation (user_inputs=user_input)
print('Output z: \n')
print(z)
a,b = fcff.build_roic_df(user_inputs=user_input)
print('Output b: \n')
print(b)




fig, ax = fcff.plot_stock_price()
plt.show()

fig, ax = fcff.plot_revenue_and_growth()
plt.show()

fig, ax = fcff.plot_ebit()
plt.show()

fig, ax = fcff.plot_operating_margin()
plt.show()

fig, ax1, ax2 = fcff.plot_invested_capital_and_roic()
plt.show()

fig, ax = fcff.plot_reinvestment_only()
plt.show()


if fcff.stock == 'NVDA':
            fcff.sankey_nvidia()
elif fcff.stock == 'MSFT': fcff.sankey_microsoft()


mc = MonteCarloInputSimulator(1000)
fig, ax = mc.plot_fair_value_distribution(
    mc.fair_values,
    stock_name='NVDA',
    stock_price= mc.fcff.bloomberg_data["stock_price"]
)
plt.show()

#Tabelle
fig, ax = mc.plot_percentile_table(mc.fair_values)
plt.show()
