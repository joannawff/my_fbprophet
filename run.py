import json
from fbprophet.serialize import model_to_json, model_from_json

# Python
import pandas as pd
from fbprophet import Prophet

# Python
df = pd.read_csv('./prophet/examples/example_wp_log_peyton_manning.csv')
df.head()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

#from fbprophet.plot import plot_plotly, plot_components_plotly

#plot_plotly(m, forecast)
#plot_components_plotly(m, forecast)

with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)  # Save model

'''
with open('serialized_model.json', 'r') as fin:
m = model_from_json(json.load(fin))  # Load model
'''
