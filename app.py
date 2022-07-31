import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash.dependencies import Output, Input
from sklearn.decomposition import PCA

from sklearn.svm import SVC
import pickle

model_file = "data/svm_model.pkl"
model = pickle.load(open(model_file, 'rb'))

data = pd.read_csv("avocado.csv")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)

data_pain = pd.read_csv("data/pain_db.csv")

label_mapping = {"level_zero": 'Base (No Pain)', "level_one": 'DELETE', "level_two": 'Minor Pain', "level_three": 'DELETE',
                 "level_four": 'Severe Pain'}

data_pain = data_pain.replace({"Label": label_mapping})
data_pain = data_pain[data_pain.Label != 'DELETE']

# means = df.groupby(['Subject-ID', 'Label']).mean()

features = ['Subject_ID', 'Label', 'CH22_Sim_corr', 'CH22_S_sd', 'CH23_A_PEAK', 'CH23_Sim_corr', 'CH23_Sim_MutInfo',
            'CH24_Sim_corr', 'CH25_meanRR', 'CH25_rmssd', 'CH26_A_PEAK', 'CH26_Sim_corr']

pain_means = data_pain.groupby(['Subject_ID', 'Label'], as_index=False).mean()

pain_means = pd.DataFrame(pain_means, columns=features)

live_data = data_pain.groupby(['Subject_ID'], as_index=False).first().drop('Label', axis=1)

# data_pain.replace([np.inf, -np.inf], np.nan, inplace=True)
# # Drop rows with NaN
# data_pain.dropna(inplace=True)

emp_g = {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        }
    }
}
global button_counter
button_counter = 0

def get_live_data(patient=-1):
    if patient == -1:
        return
    mask = (
        (live_data['Subject_ID'] == patient)
    )
    filtered_live_data = live_data.loc[mask, :]
    return filtered_live_data.to_dict('records')
    # return filtered_live_data.to_dict('records'), predict(filtered_live_data, n)


def button_check(n):
    global button_counter
    if n <= button_counter:
        return 0
    else:
        button_counter = n
        return n


def estimate(patient=-1, n=0):
    if button_check(n) == 0 or patient == -1:
        return
    mask = (
        (live_data['Subject_ID'] == patient)
    )
    filtered_live_data = live_data.loc[mask, :]
    ans = model.predict(filtered_live_data.iloc[:, 1:].to_numpy())
    if ans == 0:
        return pd.DataFrame(["Base (No Pain)"], columns=["Current Pain Intensity"]).to_dict('records')
    elif ans == 2:
        return pd.DataFrame(["Medium (Mild Pain)"], columns=["Current Pain Intensity"]).to_dict('records')
    else:
        return pd.DataFrame(["High (Severe Pain)"], columns=["Current Pain Intensity"]).to_dict('records')


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Pain Intensity Analytics"

app.layout = html.Div(style={
    'background-image': 'url("/assets/background.png")',
    'background-repeat': 'no-repeat',

},
    children=[
        html.Div(
            children=[
                html.P(children=[
                    html.Img(src=app.get_asset_url('green_monitor.png'), style={'height': '46px', 'width': '46px'})
                ], className='header-emoji'),
                html.H1(
                    children="Pain Intensity Analytics", className="header-title"
                ),
                html.P(
                    children="Analyze the awareness and pain sensitivity of patients",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Patient ID", className="menu-title"),
                        dcc.Dropdown(
                            id="patient-filter",
                            options=[
                                {"label": patient, "value": patient}
                                for patient in np.sort(data_pain['Subject_ID'].unique())
                            ],
                            # value="Albany",
                            clearable=False,
                            className="dropdown",
                            style={"width": "150%"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Type", className="menu-title"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=[
                                {"label": 'Live', "value": 'Live'},
                                {"label": 'Recorded', "value": 'Recorded'}
                            ],
                            value="Live",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                            style={"width": "150%"},
                        ),
                    ],
                ),
                html.Div(
                    style={"padding-top": "28px"},
                    children=[
                        html.Button('Calculate', id='submit-val', n_clicks=0, className="submit_button"),
                    ]
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(
                    children=dbc.Container([
                        dash_table.DataTable(
                            sort_action='native',
                            id='table',
                            css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline-table; white-space: inherit; overflow: inherit; '
                                        'text-overflow: inherit;'
                            }],
                            data=get_live_data(),
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': '#079A82',
                                'color': 'white'
                            },
                            style_data={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white'
                            },
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal'
                            },
                        )]
                    ),
                    className="card",
                ),
                html.Div(
                    children=dbc.Container([
                        dash_table.DataTable(
                            sort_action='native',
                            id='calculation',
                            css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline-table; white-space: inherit; overflow: inherit; '
                                        'text-overflow: inherit;'
                            }],
                            data=estimate(),
                            style_table={'overflowX': 'auto'},
                            style_header={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white',
                                'border-collapse' : 'collapse',
                                'border': 'none'
                            },
                            style_data={
                                'backgroundColor': 'transparent',
                                'color': 'white',
                                'border': 'none'
                            },
                            style_cell={
                                'height': '100',
                                # all three widths are needed
                                'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                                'whiteSpace': 'normal',
                                'textAlign': 'center',
                                'font_size': '26px',
                            },
                        )]
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH22_Sim_corr-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH22_S_sd-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH23_A_PEAK-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH23_Sim_corr-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH23_Sim_MutInfo-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH24_Sim_corr-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH25_meanRR-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH25_rmssd-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH26_A_PEAK-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
                html.Div(
                    children=dcc.Graph(
                        id="CH26_Sim_corr-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    [Output('table', 'data'), Output('calculation', 'data'), Output("CH22_Sim_corr-chart", "figure"),
     Output("CH22_S_sd-chart", "figure"), Output("CH23_A_PEAK-chart", "figure"), Output("CH23_Sim_corr-chart", "figure")
        , Output("CH23_Sim_MutInfo-chart", "figure"), Output("CH24_Sim_corr-chart", "figure"),
     Output("CH25_meanRR-chart", "figure"), Output("CH25_rmssd-chart", "figure"),
     Output("CH26_A_PEAK-chart", "figure"), Output("CH26_Sim_corr-chart", "figure")],
    [
        Input("patient-filter", "value"),
        Input("type-filter", "value"),
        Input('submit-val', 'n_clicks'),

    ],
)
def update_charts(patient, d_type, n_clicks):
    if d_type == 'Live':
        return get_live_data(patient), estimate(patient, n_clicks), emp_g, emp_g, emp_g, emp_g, emp_g, emp_g, emp_g, \
               emp_g, emp_g, emp_g
    mask = (
        (pain_means['Subject_ID'] == patient)
    )
    filtered_means = pain_means.loc[mask, :]
    CH22_Sim_corr_figure = {
        "data": [
            {
                "x": filtered_means['Label'],
                "y": filtered_means['CH22_Sim_corr'],
                "type": "lines",
                "hovertemplate": "%{y:.2f}<extra></extra>",
            },
        ],
        "layout": {
            "title": {
                "text": "Electromyography of Musculus Zygomaticus Major - Sim Corr",
                "x": 0.05,
                "xanchor": "left",
            },
            "xaxis": {"fixedrange": True},
            "yaxis": {"tickprefix": "", "fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    CH22_S_sd_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH22_S_sd"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electromyography of Musculus Zygomaticus Major - SSD", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#17B897"],
        },
    }

    CH23_A_PEAK_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH23_A_PEAK"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electromyography of Musculus Corrugator Supercilii - A-PEAK", "x": 0.05,
                      "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }

    CH23_Sim_corr_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH23_Sim_corr"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electromyography of Musculus Corrugator Supercilii - Sim Corr", "x": 0.05,
                      "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }

    CH23_Sim_MutInfo_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH23_Sim_MutInfo"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electromyography of Musculus Corrugator Supercilii - Similar Mutual Info", "x": 0.05,
                      "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }

    CH24_Sim_corr_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH24_Sim_corr"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electromyography of Musculus Trapezius - Sim Corr", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#2D39E1"],
        },
    }

    CH25_meanRR_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH25_meanRR"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electrocardiography - Mean RR", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D93"],
        },
    }

    CH25_rmssd_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH25_rmssd"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Electrocardiography - RM SSD", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D93"],
        },
    }

    CH26_A_PEAK_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH26_A_PEAK"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Skin Conductance Level - A-PEAK", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#2DE1D5"],
        },
    }

    CH26_Sim_corr_figure = {
        "data": [
            {
                "x": filtered_means["Label"],
                "y": filtered_means["CH26_Sim_corr"],
                "type": "lines",
            },
        ],
        "layout": {
            "title": {"text": "Skin Conductance Level - Sim Corr", "x": 0.05, "xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#2DE1D5"],
        },
    }
    return get_live_data(), estimate(), CH22_Sim_corr_figure, CH22_S_sd_figure, CH23_A_PEAK_figure, CH23_Sim_corr_figure, \
           CH23_Sim_MutInfo_figure, CH24_Sim_corr_figure, CH25_meanRR_figure, CH25_rmssd_figure, CH26_A_PEAK_figure, \
           CH26_Sim_corr_figure


if __name__ == "__main__":
    app.run_server(debug=True)
