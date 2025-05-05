import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
import os
from dash.dependencies import Input, Output


LOG_DIR = "logs"
DATA_LOG_FILE = os.path.join(LOG_DIR, "power_data.csv")
RAIN_LOG_FILE = os.path.join(LOG_DIR, "rain_log.csv")

# Load your data
power_df = pd.read_csv(DATA_LOG_FILE, parse_dates=['timestamp'])
rain_df = pd.read_csv(RAIN_LOG_FILE, parse_dates=['timestamp'])

# Drop rows with missing timestamps
power_df.dropna(subset=['timestamp'], inplace=True)
rain_df.dropna(subset=['timestamp'], inplace=True)

# Merge data on timestamp if needed
merged_df = pd.merge_asof(
    power_df.sort_values("timestamp"),
    rain_df.sort_values("timestamp"),
    on="timestamp"
)

# Create the Dash app
app = dash.Dash(__name__)

# Layout with a couple of graphs
app.layout = html.Div([
    html.H1("Sump Pump, Rain, and Soil Data Dashboard"),

    dcc.Interval(
        id='interval-component',
        interval=60*1000,
        n_intervals=0
    ),

    dcc.Graph(id='sump-pump-usage'),
    dcc.Graph(id='rainfall'),
    dcc.Graph(id='soil-moisture'),
])

def filter_last_48_hours(df, time_column='timestamp'):
    now = datetime.now()
    return df[df[time_column] >= now - timedelta(hours=48)]
    
@app.callback(
    Output('sump-pump-usage', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_sump_pump_graph(n):
    power_df = pd.read_csv(DATA_LOG_FILE, parse_dates=['timestamp'])
    power_df = filter_last_48_hours(power_df)
    fig = px.line(power_df, x='timestamp', y='power_watts',
                       title="Sump Pump Power Usage Over Time")
    return fig


@app.callback(
    Output('rainfall', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_rain_graph(n):
    rain_df = pd.read_csv(RAIN_LOG_FILE, parse_dates=['timestamp'])
    rain_df = filter_last_48_hours(rain_df)
    fig = px.line(rain_df, x='timestamp',
                       y='rainfall_mm',
                       title="24-Hour Rainfall")
    return fig


@app.callback(
    Output('soil-moisture', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_soil_moisture_graph(n):
    rain_df = pd.read_csv(RAIN_LOG_FILE, parse_dates=['timestamp'])
    rain_df = filter_last_48_hours(rain_df)
    fig = px.line(rain_df, x='timestamp',
                       y=['soil_moisture_0_to_1cm', 'soil_moisture_1_to_3cm', 'soil_moisture_3_to_9cm',
                          'soil_moisture_9_to_27cm', 'soil_moisture_27_to_81cm'],
                       title="Soil Moisture at Various Depths")
    return fig



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)
