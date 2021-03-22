import json
from fbprophet.serialize import model_to_json, model_from_json

import pandas as pd
from fbprophet import Prophet

with open('serialized_model.json', 'r') as fin:
    m = model_from_json(json.load(fin))  # Load model

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# save forcast
forecast.to_csv('forcast.csv',index=None,sep=',')

fig1=m.plot(forecast)
fig1.savefig('test_fig1.png')
fig2=m.plot_components(forecast)
fig2.savefig('test_fig2.png')

#from fbprophet.plot import plot_plotly, plot_components_plotly

#plot_plotly(m, forecast)
#plot_components_plotly(m, forecast)



'''

'''
