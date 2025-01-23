import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

class ClimateDashboard:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Climate Change Dashboard
        
        Args:
            data (pd.DataFrame): Climate change dataset
        """
        self.data = data
        self.app = dash.Dash(__name__)
        self.setup_layout()
    
    def setup_layout(self):
        """
        Configure dashboard layout
        """
        self.app.layout = html.Div([
            html.H1("Climate Change Impact Dashboard"),
            
            dcc.Graph(id='temperature-trend'),
            dcc.Graph(id='co2-emissions'),
            
            dcc.Dropdown(
                id='trend-selector',
                options=[
                    {'label': 'Global Temperature', 'value': 'Annual Mean'},
                    {'label': 'CO2 Emissions', 'value': 'Annual CO₂ emissions'}
                ],
                value='Annual Mean'
            )
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """
        Configure dashboard interactivity
        """
        @self.app.callback(
            [Output('temperature-trend', 'figure'),
             Output('co2-emissions', 'figure')],
            [Input('trend-selector', 'value')]
        )
        def update_graphs(selected_trend):
            # Temperature Trend
            temp_fig = px.line(
                self.data, x='Year', y='Annual Mean',
                title='Global Temperature Trend'
            )
            
            # CO2 Emissions
            co2_fig = px.bar(
                self.data, x='Year', y='Annual CO₂ emissions',
                title='CO2 Emissions Over Time'
            )
            
            return temp_fig, co2_fig
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """
        Run Dash server
        
        Args:
            debug (bool): Enable debug mode
            port (int): Server port
        """
        self.app.run_server(debug=debug, port=port)