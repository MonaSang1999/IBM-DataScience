# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(
                                    id='site_dropdown',
                                    options=[{'label': 'ALL SITES', 'value': 'ALL'},
                                            {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                            {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                            {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                            {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'}], 
                                    value='ALL',
                                    placeholder="Select a Launch Site here",
                                    searchable=True
                                    ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                                min=0,max=10000,step=1000,
                                                value=[min_payload,max_payload],
                                                marks={0: '0', 2500:'2500',5000:'5000',
                                                7500:'7500', 10000: '10000'}),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site_dropdown', component_property='value')
)
def get_pie_chart(site_dropdown):
    if site_dropdown == 'ALL':
        # Aggregate data for all sites
        aggregated_df = spacex_df.groupby('Launch Site')['class'].count().reset_index()
        fig = px.pie(
            aggregated_df,
            values='class', 
            names='Launch Site', 
            title='Launch Outcomes for All Sites'
        )
    else:
        # Filter data for the selected site
        filtered_df = spacex_df.loc[spacex_df['Launch Site'] == site_dropdown]
        if filtered_df.empty:
            # Handle empty data gracefully
            fig = px.pie(
                title=f'No data available for site: {site_dropdown}'
            )
        else:
            # Aggregate data for the selected site
            aggregated_df = filtered_df.groupby('class').size().reset_index(name='count')
            fig = px.pie(
                aggregated_df,
                values='count', 
                names='class', 
                title=f'Launch Outcomes for {site_dropdown}'
            )
    return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site_dropdown', component_property='value'),
    Input(component_id='payload-slider', component_property='value')])

def update_graph(entered_site, payload_slider):
    if entered_site == 'ALL':
        filtered_data = spacex_df[(spacex_df['Payload Mass (kg)']>=payload_slider[0])&(spacex_df['Payload Mass (kg)']<=payload_slider[1])]
        scatterplot = px.scatter(data_frame=filtered_data, x="Payload Mass (kg)", y="class", 
        color="Booster Version Category")
        return scatterplot
    else:
        specific_df=spacex_df[spacex_df['Launch Site'] == entered_site]
        filtered_data = specific_df[(specific_df['Payload Mass (kg)']>=payload_slider[0])&(spacex_df['Payload Mass (kg)']<=payload_slider[1])]
        scatterplot = px.scatter(data_frame=filtered_data, x="Payload Mass (kg)", y="class", 
        color="Booster Version Category")
        return scatterplot

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)


#python3.11 spacex_dash_app.py