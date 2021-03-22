import json
from fbprophet.serialize import model_to_json, model_from_json
from fbprophet.plot import plot_plotly, plot_components_plotly

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
fig1.savefig('train_fig1.png')
fig2 = m.plot_components(forecast)
fig2.savefig('train_fig2.png')

plot_plotly(m, forecast)
plot_components_plotly(m, forecast)

with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)  # Save model

