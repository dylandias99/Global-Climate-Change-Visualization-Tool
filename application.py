import io
import dash
import dash_table
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objs as gobj
import pymongo
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.arima.model as stats
import warnings

warnings.filterwarnings("ignore")
buffer = io.StringIO()
pd.options.plotting.backend = "plotly"

def countries():
    countries = pd.read_csv("./Climate_Change_Countries/Countries.csv")
    all_countries_name = list(countries['Countries'])
    return all_countries_name

def temp_countries():
    countries = pd.read_csv("./Climate_Change_Countries/Temp_Countries.csv")
    all_countries_name = list(countries['Countries'])
    return all_countries_name

def fossil_fuel(value: str):
    client = pymongo.MongoClient(
        "mongodb+srv://Dylan_Dias:Mongodbatlas@cluster0.hszkc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = client["climate_change"]
    fossil_fuel = "fossil_fuel_" + value
    fuel = db[fossil_fuel]
    return pd.DataFrame(list(fuel.find()))

def temperature_data():
  client = pymongo.MongoClient(
    "mongodb+srv://Dylan_Dias:Mongodbatlas@cluster0.hszkc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
  db = client["climate_change"]
  time_series = "time_series_temp"
  temp = db[time_series]
  df = pd.DataFrame(list(temp.find()))
  df['Years'] = pd.to_datetime(df['Years'], format='%Y')
  return df
def temperature_data():
  client = pymongo.MongoClient(
    "mongodb+srv://Dylan_Dias:Mongodbatlas@cluster0.hszkc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
  db = client["climate_change"]
  time_series = "time_series_temp"
  temp = db[time_series]
  df = pd.DataFrame(list(temp.find()))
  df['Years'] = pd.to_datetime(df['Years'], format='%Y')
  return df

def temp_country_line(country):
    temp_data = temperature_data().sort_values(by="Years")
    line_country = px.line(height=400)

    for i in range(0, len(country)):
        line_country.add_trace(
            go.Scatter(x=temp_data["Years"], y=temp_data[country[i]], name=country[i], mode='lines+markers'))
    line_country.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},legend_title_text='Countries',
                      yaxis_title="Temperature °Celsius")

    return line_country

def world_temp():
    temp_data = temperature_data().sort_values(by="Years")
    world_line = px.line(temp_data, x=temp_data["Years"], y=temp_data["World"], text=temp_data["World"], height=300)
    world_line.update_traces(textposition="bottom right")
    world_line.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      yaxis_title="Temperature °Celsius")
    return world_line

def climate_change_choropleth(climate, year):
    new = climate[["Countries", str(year)]]
    data = dict(type='choropleth',
                locations=new["Countries"],
                locationmode='country names',
                autocolorscale=False,
                colorscale='Blues',  # ["red", "white", "yellow","blue"],
                marker_line_color='darkgray',
                text=new.iloc[:, 0],
                z=new.iloc[:, 1])
    layout = dict(geo=dict(scope='world'))
    worldmap = gobj.Figure(data=[data], layout=layout)
    worldmap.update_geos(projection_type="robinson")
    worldmap.update_layout(margin={"r":0,"t":20,"l":0,"b":20}, legend_title_text='Metric tons of carbon dioxide')
    return worldmap

def join_fossil():
    coal = fossil_fuel("coal")
    oil = fossil_fuel("oil")
    gas = fossil_fuel("gas")

    new_coal = coal[["Countries", "Total_Coal_Emission"]]
    new_oil = oil[["Countries", "Total_Oil_Emission"]]
    new_gas = gas[["Countries", "Total_Gas_Emission"]]

    total_pol_1 = new_coal.join(new_oil.set_index("Countries"), on="Countries")
    total_pol_2 = total_pol_1.join(new_gas.set_index("Countries"), on="Countries")
    return total_pol_2

def fossil_fuel_radar(total_data, countries):
    fossil_countries = total_data[total_data['Countries'].isin(countries)]
    stats = fossil_countries.sort_values(by=['Countries'])
    stats_list = stats[['Total_Coal_Emission', 'Total_Oil_Emission', 'Total_Gas_Emission']].values.tolist()
    Attribute = ["Total MtCo2 in Coal", "Total MtCo2 in Oil", "Total MtCo2 in Gas"]
    figure_3 = px.line_polar(line_close=True, height=500)
    figure_3.update_layout(legend_title_text='Countries', font_size=12, polar=dict(bgcolor="#F0F8FF", angularaxis=dict(
        gridcolor='#40E0D0'
    ), radialaxis=dict(gridcolor="#40E0D0", linecolor="#40E0D0")), font=dict(
        family="'Verlag', sans-serif",
        color="#003399"
    ),
                           paper_bgcolor="white")

    team_name = sorted(countries)
    for i in range(0, len(stats_list)):
        figure_3.add_trace(go.Scatterpolar(
            r=stats_list[i],
            theta=Attribute,
            fill='toself',
            name=team_name[i]
        ))
        i += 1
    return figure_3

def time_series_fossil(value: str):
  client = pymongo.MongoClient(
    "mongodb+srv://Dylan_Dias:Mongodbatlas@cluster0.hszkc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
  db = client["climate_change"]
  time_series = "time_series_" + value
  fossil = db[time_series]
  return pd.DataFrame(list(fossil.find()))

def join_time_series_fossil():
    coal = time_series_fossil("coal")
    oil = time_series_fossil("oil")
    gas = time_series_fossil("gas")

    new_coal = coal[["Years", "Total_Coal_Emission"]]
    new_oil = oil[["Years", "Total_Oil_Emission"]]
    new_gas = gas[["Years", "Total_Gas_Emission"]]

    total_pol_1 = new_coal.join(new_oil.set_index("Years"), on="Years")
    total_pol_2 = total_pol_1.join(new_gas.set_index("Years"), on="Years")
    return total_pol_2

def time_series_area_graph(ran):
    total = join_time_series_fossil().sort_values(by='Years')
    new_ran = []
    for i in range(ran[0], ran[1] + 1):
        new_ran.append(str(i))

    fig = px.area()
    fig.add_trace(go.Scatter(x=new_ran, y=total["Total_Coal_Emission"], fill='tozeroy', name="Coal", mode='none'))
    fig.add_trace(go.Scatter(x=new_ran, y=total["Total_Oil_Emission"], fill='tozeroy', name="Oil", mode="none"))
    fig.add_trace(go.Scatter(x=new_ran, y=total["Total_Gas_Emission"], fill='tozeroy', name="Gas", mode='none'))
    fig.update_layout(margin={"r":0,"t":10,"l":0,"b": 0},
                      yaxis_title="Metric tons of carbon dioxide",
                      )

    return fig

def line_chart_countries(ran, country, fossilfuel):
    fuel = time_series_fossil(fossilfuel).sort_values(by='Years')
    new_ran = []
    for i in range(ran[0], ran[1] + 1):
        new_ran.append(str(i))

    fig = px.line(height=400)

    for i in range(0, len(country)):
        fig.add_trace(go.Scatter(x=new_ran, y=fuel[country[i]], name=country[i], mode='lines+markers'))

    fig.update_layout(margin={"r":0,"t":5,"l":0,"b": 0},legend_title_text='Countries',
                      yaxis_title="Metric tons of carbon dioxide")
    return fig

def temperature_change():
  client = pymongo.MongoClient(
    "mongodb+srv://Dylan_Dias:Mongodbatlas@cluster0.hszkc.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
  db = client["climate_change"]
  time_series = "temp_change"
  temp = db[time_series]
  return pd.DataFrame(list(temp.find()))

def world_temp_geo(year):
    total = temperature_change()
    world_data = total[total["Months"] == "Yearly"]
    new = world_data[["Area", str(year)]]
    data = dict(type='choropleth',
                    locations=new["Area"],
                    locationmode='country names',
                    autocolorscale=False,
                    colorscale='Reds',  # ["red", "white", "yellow","blue"],
                    marker_line_color='darkgray',
                    text=new.iloc[:, 0],
                    z=new.iloc[:, 1])
    layout = dict(geo=dict(scope='world'))
    worldmap = gobj.Figure(data=[data], layout=layout)
    worldmap.update_geos(projection_type="robinson")
    worldmap.update_layout(margin={"r":0,"t":20,"l":0,"b":20}, legend_title_text='Temperature °Celsius')
    return worldmap

def exp_country_violin(country, type):
    temp = time_series_fossil(type)
    fig = go.Figure()
    for i in range(0 , len(country)):
        fig.add_trace(go.Violin(y=temp[country[i]],
                            box_visible=True,
                            meanline_visible=True,
                            name=country[i],
                            points='all'))
    fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 40},legend_title_text='Countries', height=350,
                      font=dict(
                                family="Verlag",
                                size=15,
                                color="#073980"
                            ),
                      yaxis_title="Metric tons of carbon dioxide")
    return fig


def temp_country_violin(country):
    temp = temperature_data()
    fig = go.Figure()
    for i in range(0 , len(country)):
        fig.add_trace(go.Violin(y=temp[country[i]],
                            box_visible=True,
                            meanline_visible=True,
                            name=country[i],
                            points='all'))
    fig.update_layout(margin={"r": 0, "t": 10, "l": 0, "b": 40},legend_title_text='Countries',
                      yaxis_title="Temperature °Celsius")
    return fig

def data_description(data_type):
  fossil = time_series_fossil(data_type)
  fossil_1 = fossil.describe()
  new_col = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
  fossil_1.insert(loc=0, column='Description', value=new_col)
  return fossil_1

def temp_data_description():
  temp = temperature_data()
  temp_1 = temp.describe()
  new_col = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
  temp_1.insert(loc=0, column='Description', value=new_col)
  return temp_1

def exp_vis(type):
    data = fossil_fuel(type)

    count = data[data.iloc[:, -1] == 0]
    count_1 = data[data.iloc[:, -1] > 0]

    range_1 = data[((data.iloc[:, -1] >= 0) & (data.iloc[:, -1] <= 500))]
    range_2 = data[((data.iloc[:, -1] > 500) & (data.iloc[:, -1] <= 5000))]
    range_3 = data[(data.iloc[:, -1] > 5000)]

    d_sort = data.sort_values(by=data.columns[-1], ascending=False)
    top_10 = d_sort.head(10)

    countries = top_10["Countries"].values.tolist()
    ran = [2000, 2019]

    count_bar = go.Figure(data=[
        go.Bar(x=["Countries with 0 fossil fuel emission", "Countries with some fossil fuel emission"], y=[count["Countries"].count(), count_1["Countries"].count()],
               marker_color=["#1E90FF", "#2a52be"],
               width=[0.5, 0.5])
    ])

    count_bar.update_layout(margin={"r":0, "t": 0, "l": 0, "b": 0},
                      title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                      yaxis_title="Count",
                      font=dict(
                                family="Verlag",
                                size=15,
                                color="#073980"
                            ),
                      height=290
                      )

    range_bar = go.Figure(data=[
        go.Bar(x=["Less than 500 MtCo2", "500 to 5000 MtCo2", "More than 5000 MtCo2"],
               y=[range_1["Countries"].count(), range_2["Countries"].count(), range_3["Countries"].count()],
               marker_color=[ "#1E90FF", "#2a52be","#37536d"],
               width=[0.6, 0.6, 0.6])
    ])

    range_bar.update_layout(margin={"r": 50, "t": 0, "l": 0, "b": 0},
                      title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                      yaxis_title="Count",
                      font=dict(
                                family="Verlag",
                                size=15,
                                color="#073980"
                            ),
                      height=290
                      )

    top = px.bar(top_10, x=top_10.columns[-2], y=top_10.columns[-1],
                 title="Top 10 countries with highest carbon dioxide emissions for " + type + " fossil fuel type")
    top.update_traces(marker_color="#1E90FF")
    top.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0},
                      title={
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'},
                      font=dict(
                                family="Verlag",
                                size=12,
                                color="#073980"
                            )
                      )

    top_line = line_chart_countries(ran, countries, type)
    top_line.update_layout(
                      font=dict(
                                family="Verlag",
                                size=12,
                                color="#073980"
                            ))
    return count_bar, range_bar, top, top_line

def top_temp_bar(years):
    tem = temperature_change()
    year = str(years)

    rem_tem = tem[~tem["Area"].isin(["World",'Africa','Eastern Africa','Middle Africa','Northern Africa','Southern Africa','Western Africa','Americas','Northern America','Central America','Caribbean','South America','Asia','Central Asia','Eastern Asia','Southern Asia','South-Eastern Asia','Western Asia','Europe','Northern Europe','Eastern Europe','Southern Europe','Western Europe','Oceania','Australia', 'New Zealand','Melanesia','Micronesia','Polynesia','European Union'])].sort_values(by=year, ascending=False)

    new_data = rem_tem[rem_tem["Months"] == "Yearly"].sort_values(by=year, ascending=False)

    top_temp = new_data[["Area", year]].head(10)

    top_tem = go.Figure(data=[
            go.Bar(x=top_temp["Area"],
                   y=top_temp[year])
        ])
    top_tem.update_traces(marker_color="#1E90FF")
    top_tem.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          font=dict(
                                    family="Verlag",
                                    size=12,
                                    color="#073980"
                                ),
                      yaxis_title="Temperature °Celsius")
    return top_tem

def cluster_analysis():
    scaler = MinMaxScaler()
    total_co2 = join_fossil()
    total_co2_display = join_fossil()
    scaler.fit(total_co2[['Total_Coal_Emission']])
    total_co2['Total_Coal_Emission'] = scaler.transform(total_co2[['Total_Coal_Emission']])
    scaler.fit(total_co2[['Total_Oil_Emission']])
    total_co2['Total_Oil_Emission'] = scaler.transform(total_co2[['Total_Oil_Emission']])
    scaler.fit(total_co2[['Total_Gas_Emission']])
    total_co2['Total_Gas_Emission'] = scaler.transform(total_co2[['Total_Gas_Emission']])
    sse = []
    k_rng = range(1, 30)
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(total_co2[['Total_Coal_Emission', "Total_Oil_Emission", "Total_Gas_Emission"]])
        sse.append(km.inertia_)

    elbow_plot = px.line(x=k_rng, y=sse, labels=dict(x="Number of Clusters", y="SSE"), title="Elbow Method")
    elbow_plot.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0},height=400)
    km = KMeans(n_clusters=5)

    y_predicted = km.fit_predict(total_co2[['Total_Coal_Emission', "Total_Oil_Emission", "Total_Gas_Emission"]])

    total_co2_display['cluster'] = y_predicted
    cluster_plot = px.scatter_3d(
    total_co2_display, x=total_co2_display['Total_Coal_Emission'], y=total_co2_display['Total_Oil_Emission'], z=total_co2_display['Total_Gas_Emission'], color=total_co2_display['cluster'], hover_name=total_co2_display["Countries"], height=500)
    cluster_plot.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return elbow_plot, cluster_plot

def arima_model():
    data_set = temperature_data()
    read = data_set[["Years", "World"]].sort_values(by='Years')
    read.set_index("Years", inplace=True)

    train = read.iloc[:-9]
    test = read.iloc[-9:]

    model = ARIMA(train["World"], order=(10, 1, 0))
    model_fit = model.fit()
    summ_of_model = model_fit.summary()

    start = len(train)
    end = len(train) + len(test) - 1
    train_pred = model_fit.predict(start=start, end=end, typ="levels")

    model2 = stats.ARIMA(read["World"], order=(9, 1, 0))
    model_fit2 = model2.fit()

    final_pred = model_fit2.predict(start=len(read), end=len(read) + 10, typ='levels').rename('ARIMA Predictions')

    train_line = pd.DataFrame(train_pred)
    train_line.reset_index(inplace=True)

    final = pd.DataFrame(final_pred)
    final.reset_index(inplace=True)

    viz_set = temperature_data()
    viz = viz_set[["Years", "World"]].sort_values(by='Years')

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=viz["Years"], y=viz["World"],
                             mode='lines', name="Actual"))
    fig1.add_trace(go.Scatter(x=train_line["index"], y=train_line["predicted_mean"],
                             mode='lines', name="Testing"))
    fig1.update_layout(yaxis_title="Temperature °Celsius")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=viz["Years"], y=viz["World"],
                             mode='lines', name="Observed"))
    fig2.add_trace(go.Scatter(x=final["index"], y=final["ARIMA Predictions"],
                    mode='lines', name="ARIMA Predictions"))
    fig2.update_layout(yaxis_title="Temperature °Celsius")
    return fig1, fig2

content = html.Div(id="page-content", children=["content"], className="climate_content")

sidebar = html.Div([
                    dbc.Nav([
                        html.Div([
                            dbc.NavLink("Introduction", href="/", active="exact", className="links"),
                            dbc.NavLink("Dashboard ", href="/visualizations", className='links', active="exact"),
                            dbc.NavLink("Exploratory Analysis", href="/exploratory_analysis", className='links', active="exact"),
                            dbc.NavLink("Statistical Models", href="/analysis", className='links', active="exact"),
                        ], className='nav_links')
                    ], vertical=True, pills=True, className='climate_nav_bar'),
                    ])

footer = html.Div([
               html.Div("© Created By Dylan Dias"),
               html.Div("Under the guidance and supervision of Prof Tae Oh and Prof Michael McQuaid")
], className="climate_footer")

app = dash.Dash(__name__)
# Beanstalk looks for application by default, if this isn't set you will get a WSGI error.
application = app.server
app.title = 'Climate Change Analysis'
app.layout = html.Div(children=[
     				dcc.Location(id="url"),
                    html.Div(html.H1(["Global Climate Change Analysis"],), className="climate_header"),
                    sidebar,
                    content,
                    footer,

], className="climate_layout")

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
            html.Div([
                html.Div("About the Project", className="intro_item_1 intro_title"),
                html.Div("The main goal of this project is to analyze the Earth’s climate over the recent years and predicting future climate change based on human activities."
                         " Through this interface we will be able understand what all changes are taking place in the Earth’s atmosphere, implementing a time-series model to predict future climate change if this pattern continues and clustering countries based on their carbon emission level."
                         , className="intro_item_2"),
                html.Div("Climate Change", className="intro_item_3 intro_title"),
                html.Div([html.Div("What is climate change?",className="text"),
                          html.Div([
                              html.Div(
                                  "Climate change is a change in the usual weather found in a place."
                                  " This could be a change in how much rain a place usually gets in a year."
                                  " Or it could be a change in a place's usual temperature for a month or season."
                                  " Climate change is also a change in Earth's climate. This could be a change in Earth's usual temperature."
                                  " Or it could be a change in where rain and snow usually fall on Earth. "
                                  " Weather can change in just a few hours. Climate takes hundreds or even millions of years to change.")
                          ], className="overlay overlay_text_1"),
                          html.Div(html.Img(src='assets/person.jpeg', className="img"))
                          ]
                         , className="intro_hover intro_item_4"),
                html.Div([html.Div("What are its causes?", className="text"),
                          html.Div([
                              html.Div(
                                  "Many things can cause climate to change all on its own."
                                  " Earth's distance from the sun can change."
                                  " The sun can send out more or less energy."
                                  " Most scientists say that humans can change climate too."
                                  " People drive cars."
                                  " People heat and cool their houses."
                                  " People cook food."
                                  " All those things take energy."
                                  " One way we get energy is by burning coal, oil and gas."
                                  " Burning these things puts gases into the air."
                                  " The gases cause the air to heat up."
                                  " This can change the climate of a place."
                                  " It also can change Earth's climate.")
                          ], className="overlay overlay_text_2"),
                          html.Div(html.Img(src='assets/global.jpg', className="img"))

                          ]
                         , className="intro_hover intro_item_5"),
                html.Div([html.Div("What are its effects?", className="text"),
                          html.Div([
                              html.Div(
                                  "Scientists have high confidence that global temperatures will continue to rise for decades to come, largely due to greenhouse gases produced by human activities."
                                  " This would cause more snow and ice to melt."
                                  " Oceans would rise higher."
                                  " Some places would get hotter."
                                  " Other places might have colder winters with more snow."
                                  " Some places might get more rain."
                                  " Other places might get less rain."
                                  " Some places might have stronger hurricanes.")
                          ], className="overlay overlay_text_3"),
                          html.Div(html.Img(src='assets/effects.jpeg', className="img"))
                          ]
                         , className="intro_hover intro_item_6"),
                html.Div("Fossil Fuels", className="intro_item_7 intro_title"),
                html.Div("Coal, crude oil, and natural gas are all considered fossil fuels because they were formed from the fossilized, buried remains of plants and animals that lived millions of years ago."
                         " Because of their origins, fossil fuels have a high carbon content.", className="intro_item_8"),
                html.Div("Types of fossil fuel", className="intro_item_9 intro_title"),
                html.Div([
                          html.Div([
                              html.Div(
                                  "Crude oil, or petroleum is a liquid fossil fuel made up mostly of hydrocarbons."
                                  " Oil can be found in underground reservoirs; in the cracks, crevices, and pores of sedimentary rock; or in tar sands near the earth’s surface."
                                  " It’s accessed by drilling, on land or at sea, or by strip mining in the case of tar sands oil and oil shale."
                              )], className="overlay overlay_text_4"),
                          html.Div(html.Img(src='assets/oil_1.jpeg', className="img_1"))

                          ]
                         , className="intro_hover intro_item_10"),
                html.Div([
                          html.Div([
                              html.Div(
                                  "Coal is a solid, carbon-heavy rock that comes in four main varieties differentiated largely by carbon content: lignite, sub-bituminous, bituminous and anthracite."
                                  " Regardless of variety, however, all coal is dirty."
                                  " In terms of emissions, it’s the most carbon-intensive fossil fuel we can burn."
                                  )
                          ], className="overlay overlay_text_5"),
                          html.Div(html.Img(src='assets/coal_1.png', className="img_1"))

                          ]
                         , className="intro_hover intro_item_11"),
                html.Div([html.Div([
                              html.Div(
                                 "Natural gas is a fossil energy source that formed deep beneath the earth's surface."
                                 " Natural gas contains many different compounds."
                                 " Natural gas also contains smaller amounts of natural gas liquids, and non-hydrocarbon gases, such as carbon dioxide and water vapor."
                                 " We use natural gas as a fuel and to make materials and chemicals."
                                 )
                          ], className="overlay overlay_text_6"),
                          html.Div(html.Img(src='assets/gas_1.jpeg', className="img_1"))
                          ]
                         , className="intro_hover intro_item_12"),
                html.Div("Effects of Fossil Fuels", className="intro_item_13 intro_title"),
                html.Ul(
                    [
                        html.H3("Global warming pollution"),
                        html.Li("When we burn oil, coal, and gas, we don’t just meet our energy needs—we drive the current global warming crisis as well."),
                        html.Li("Fossil fuels produce large quantities of carbon dioxide when burned. Carbon emissions trap heat in the atmosphere and lead to climate change."),
                        html.Li("In the United States, the burning of fossil fuels, particularly for the power and transportation sectors, accounts for about three-quarters of our carbon emissions.")
                    ], className="intro_item_14"
                ),
                html.Ul(
                    [
                        html.H3("Air pollution"),
                        html.Li("Fossil fuels emit more than just carbon dioxide when burned."),
                        html.Li("Fossil fuel–powered cars, trucks, and boats are the main contributors of poisonous carbon monoxide and nitrogen oxide, which produces smog (and respiratory illnesses) on hot days.")
                    ], className="intro_item_15"
                ),
                html.Ul(
                    [
                        html.H3("Ocean acidification"),
                        html.Li("When we burn oil, coal, and gas, we change the ocean’s basic chemistry, making it more acidic."),
                        html.Li("Our seas absorb as much as a quarter of all man-made carbon emissions."),
                        html.Li("Since the start of the Industrial Revolution (and our coal-burning ways), the ocean has become 30 percent more acidic."),
                        html.Li("As the acidity in our waters goes up, the amount of calcium carbonate—a substance used by oysters, lobsters, and countless other marine organisms to form shells—goes down."),
                        html.Li("This can slow growth rates, weaken shells, and imperil entire food chains.")
                    ], className="intro_item_16"
                )
            ], className="intro_content")
        ]
    elif pathname == "/visualizations":
        return [
            html.Div([
            html.Div([
                html.Div("Select data type:", className="data_type_title"),
                html.Div(dcc.RadioItems(
                id= "data_type_radio_button",
                options=[
                    {'label': ' Co2 ', 'value': 'co2'},
                    {'label': ' Temperature ', 'value': 'temp'},
                ] ,
                labelStyle={
                    'display': 'inline-block',
                    'padding': '0 0 0 0.4em',
                },
                value='co2'
                 , className="data_type_radio"))
            ], className="data_type_layout"),
            html.Div(id="data_type_vis_output")]
            )
        ]
    elif pathname == "/exploratory_analysis":
        return [
            html.Div([
                html.Div([
                    html.Div("Select data type:", className="exp_type_title"),
                    html.Div(dcc.RadioItems(
                        id="exp_type_radio_button",
                        options=[
                            {'label': ' Co2 ', 'value': 'co2'},
                            {'label': ' Temperature ', 'value': 'temp'},
                        ],
                        labelStyle={
                            'display': 'inline-block',
                            'padding': '0 0 0 0.4em',
                        },
                        value='co2'
                        , className="exp_type_radio"))
                ], className="exp_type_layout"),
                html.Div(id="exp_type_vis_output")]
            )
        ]
    elif pathname == "/analysis":
        return [html.Div([
                html.Div([html.Div(["Statistical Models:"], className="stat_model_title"),
                html.Div(dcc.RadioItems(

                id= "stats_radio_button",
                options=[
                    {'label': ' Clustering Model', 'value': 'cluster'},
                    {'label': ' Forecasting Model', 'value': 'forecast'},
                ],
                value='cluster'
                 , className="stat_radio"))
                ], className="stat_models_main"),
            html.Div(id="stat_models")
        ], className="stat_radio_buttons_output")
        ]

@app.callback(
    Output('exp_type_vis_output', 'children'),
    Input('exp_type_radio_button', 'value'),
)
def exp_analysis(exp_type):
    continent = ["Asia", "Africa", "Europe", "Northern America", "South America", "Australia", "Antarctica"]
    temp_line = temp_country_line(continent)
    temp_line.update_layout(
                            font=dict(
                                family="Verlag",
                                size=15,
                                color="#073980"
                            ))
    temp_violine = temp_country_violin(continent)
    temp_violine.update_layout(height= 420,
                               font=dict(
                                family="Verlag",
                                size=15,
                                color="#073980"
                            ))
    decs = temp_data_description()

    all_countries = countries()
    if exp_type == "co2":
        content = [
            html.Div([
                html.Div(["Data Description"], className="exp_co2_item_1 exp_co2_title"),
                html.Div(["The Co2 dataset consists of 3 csv file each for coal, oil and gas fossil fuel types."
                          " Each csv file contains 223 columns and 20 rows."
                          " Float and int are the two data types present in these csv files."],
                         className="exp_co2_item_2"),
                html.Div(
                    dcc.RadioItems(
                        id="dataset_fuel_type",
                        options=[
                            {'label': ' Gas', 'value': 'gas'},
                            {'label': ' Coal', 'value': 'coal'},
                            {'label': ' Oil', 'value': 'oil'}
                        ],
                        value='gas',
                        labelStyle={'display': 'inline-block',
                                    'color': '#003399',
                                    'padding': '0 0 0 1em',
                                    'font-size': '1.3em'},
                    ), className="exp_co2_item_3"),
                html.Div(id="exp_co2_data_description", className="exp_co2_item_4"),
                html.Div(["Data Visualization Analysis"], className="exp_co2_item_5 exp_co2_title"),
                html.Div(
                    dcc.RadioItems(
                        id="dataset_fuel_type_1",
                        options=[
                            {'label': ' Gas', 'value': 'gas'},
                            {'label': ' Coal', 'value': 'coal'},
                            {'label': ' Oil', 'value': 'oil'}
                        ],
                        value='gas',
                        labelStyle={'display': 'inline-block',
                                    'color': '#003399',
                                    'padding': '0 0 0 1em',
                                    'font-size': '1.3em'},
                    ), className="exp_co2_item_6"),
                html.Div(["The violin plot below shows statistical distribution of numerical data."],
                         className="exp_co2_item_7"),
                html.Div(["Select countries to view violin plot"], className="exp_co2_item_8"),
                html.Div(dcc.Dropdown(
                    id='countries_dropdown_2',
                    options=[
                        {'label': i, 'value': i} for i in all_countries
                    ],
                    value=["India", "Pakistan", "South Korea"],
                    multi=True,
                    className="country_dropdown_content_2"
                ),className="exp_co2_item_9"),
                html.Div(id="exp_violin_plot", className="exp_co2_item_10"),
                html.Div(["The bar plot on the left shows the count of countries with zero co2 emission and some co2 emission for the selected fossil fuel type. The bar plot on the right shows the count of countries for a specific Co2 range."], className="exp_co2_item_11"),
                html.Div(id="count_plot", className="exp_co2_item_12"),
                html.Div(id="count_plot_1", className="exp_co2_item_13"),
                html.Div(["The bar plot on the left shows the top 10 coutries with the highest carbon dioxide emission level and line plot on the right show the emission level from 2000 - 2019 for those 10 countries for the selected fossil fuel type."],
                         className="exp_co2_item_14"),
                html.Div(id="count_plot_2", className="exp_co2_item_15"),
                html.Div(id="count_plot_3", className="exp_co2_item_16"),
            ], className="exp_co2_analysis")
        ]
    elif exp_type == "temp":
        content = [
            html.Div([
                html.Div(["Data Description"], className="exp_temp_item_1 exp_temp_title"),
                html.Div(["The temperature dataset consists of a single csv file for temperature change data for various countires, major cities and continents."
                          " Each csv file contains 274 columns and 20 rows."
                          " Float and int are the two data types present in these csv files."],
                         className="exp_temp_item_2"),
                html.Div([
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i} for i in decs.columns],
                        data=decs.to_dict('records'),
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#1877F2',
                                'color': 'white'
                            },
                            {
                                'if': {'row_index': 'even'},
                                'backgroundColor': '#1877F2',
                                'color': 'white'
                            }
                        ],
                        style_cell={'padding': '5px'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'Country'},
                             'width': '6em'},
                            {'if': {'column_id': 'Pts'},
                             'width': '4em'},
                        ],
                        style_data={
                            'font-family': '"Verlag", sans-serif',
                            'text-align': 'center',
                            'font-size': '12px',
                            'padding': '6px',
                        },
                        style_table={
                            'overflowX': 'auto',
                            'overflowY': 'auto',
                            'width': '71em'
                        },
                        style_header={
                            'font-family': '"Verlag", sans-serif',
                            'backgroundColor': '#002D72',
                            'color': 'white',
                            'text-align': 'center',
                            'font-size': '15px'
                        }
                    )
                ], className="exp_temp_item_3"),
                html.Div(["Data Visualization Analysis"], className="exp_temp_item_4 exp_temp_title"),
                html.Div(["The violin plot below shows statistical distribution of numerical data for all the continents."],
                         className="exp_temp_item_5"),
                html.Div(dcc.Graph(figure=temp_violine), className="exp_temp_item_6"),
                html.Div(["The line plot below shows the temperature change happening from 2000 - 2019 for all the continents."],
                         className="exp_temp_item_7"),
                html.Div(dcc.Graph(figure=temp_line), className="exp_temp_item_8"),
                html.Div(["The bar plot below shows the top 10 hottest coutries for the selected year:"], className="exp_temp_item_9"),
                html.Div([
                    dcc.Dropdown(
                        id='year_dropdown_temp',
                        options=[
                            {'label': '2000', 'value': "2000"},
                            {'label': '2001', 'value': "2001"},
                            {'label': '2002', 'value': "2002"},
                            {'label': '2003', 'value': "2003"},
                            {'label': '2004', 'value': "2004"},
                            {'label': '2005', 'value': "2005"},
                            {'label': '2006', 'value': "2006"},
                            {'label': '2007', 'value': "2007"},
                            {'label': '2008', 'value': "2008"},
                            {'label': '2009', 'value': "2009"},
                            {'label': '2010', 'value': "2010"},
                            {'label': '2011', 'value': "2011"},
                            {'label': '2012', 'value': "2012"},
                            {'label': '2013', 'value': "2013"},
                            {'label': '2014', 'value': "2014"},
                            {'label': '2015', 'value': "2015"},
                            {'label': '2016', 'value': "2016"},
                            {'label': '2017', 'value': "2017"},
                            {'label': '2018', 'value': "2018"},
                            {'label': '2019', 'value': "2019"}
                        ],
                        value="2001",
                        className="year_dropdown_temp"
                    )
                ], className="exp_temp_item_10"),
                html.Div(id="exp_temp_top_line", className="exp_temp_item_11")
            ], className="exp_temp_analysis")
        ]
    else:
        content = "Error"
    return html.Div(content)

@app.callback(
    Output('data_type_vis_output', 'children'),
    Input('data_type_radio_button', 'value'),
)
def stat_models(value):
    temp_world = world_temp()
    all_countries = countries()
    all_temp_countries = temp_countries()
    if value == "co2":
        content = [
    html.Div([
    html.Div([html.Span(["Select Country & Year"], className="country_year"),
    dcc.Dropdown(
        id='year_dropdown',
    options=[
    {'label': '2000', 'value': "2000"},
    {'label': '2001', 'value': "2001"},
    {'label': '2002', 'value': "2002"},
    {'label': '2003', 'value': "2003"},
    {'label': '2004', 'value': "2004"},
    {'label': '2005', 'value': "2005"},
    {'label': '2006', 'value': "2006"},
    {'label': '2007', 'value': "2007"},
    {'label': '2008', 'value': "2008"},
    {'label': '2009', 'value': "2009"},
    {'label': '2010', 'value': "2010"},
    {'label': '2011', 'value': "2011"},
    {'label': '2012', 'value': "2012"},
    {'label': '2013', 'value': "2013"},
    {'label': '2014', 'value': "2014"},
    {'label': '2015', 'value': "2015"},
    {'label': '2016', 'value': "2016"},
    {'label': '2017', 'value': "2017"},
    {'label': '2018', 'value': "2018"},
    {'label': '2019', 'value': "2019"}
    ],
    value="2000",
    className= "year_dropdown_content"
    ),
    dcc.Dropdown(
        id='countries_dropdown',
    options=[
        {'label': i, 'value': i} for i in all_countries
    ],
    value="India",
    className= "country_dropdown_content"
    )
    ], className="visualization_item_1 visual_box"),
    html.Div(id='climate_oil_value', className="visualization_item_2 visual_box_1"),
    html.Div(id='climate_coal_value', className="visualization_item_3 visual_box_2"),
    html.Div(id='climate_gas_value', className="visualization_item_4 visual_box_3"),
     html.Div(html.Span(["Select Fossil Fuel Type & Year"], className="fuel_year"),className="visualization_item_5"),
    html.Div(dcc.RadioItems(
    id="fossil_fuel_type",
    options=[
            {'label': ' Gas', 'value': 'gas'},
            {'label': ' Coal', 'value': 'coal'},
            {'label': ' Oil', 'value': 'oil'}
    ],
    value='gas',
     labelStyle={'display': 'inline-block',
                 'color':'#003399',
                 'padding':'0 0 0 1em',
                 'font-size':'1.3em'}
    ), className="visualization_item_6 fossil_fuels"),
    html.Div(id="geo_plot_output", className="visualization_item_7"),
    html.Div([
        dcc.Slider(
            id="fossil_fuels_year",
            min=2000,
            max=2019,
            marks={
                2000: '2000',
                2001: '2001',
                2002: '2002',
                2003: '2003',
                2004: '2004',
                2005: '2005',
                2006: '2006',
                2007: '2007',
                2008: '2008',
                2009: '2009',
                2010: '2010',
                2011: '2011',
                2012: '2012',
                2013: '2013',
                2014: '2014',
                2015: '2015',
                2016: '2016',
                2017: '2017',
                2018: '2018',
                2019: '2019'
            },
            value=2000,
            included=False,
            className="slider"
        )
    ], className="visualization_item_8 climate_years"),
    html.Div([
    html.Div(["Total Fossil Fuel Emission From 2000-2019 By Each Country"], className="polar_chart_title"),
    html.Div(id='polar_countries', className="polar_countries", style={
        "margin":"-0.5em 0 0 0"
    }),
    dcc.Dropdown(
        id='polar_countries_dropdown',
    options=[
        {'label': i, 'value': i} for i in all_countries
    ],
    multi=True,
    value=["Denmark", "Argentina", "Belgiu"],
    className= "polar_countries_dropdown",
        style={
            "margin":"-2em 0 0 0"
        }
    )
    ], className='visualization_item_9 fossil_fuel_radar'),
    html.Div([
        html.Div(["World Fossil Fuel Emission"], className="area_chart_title"),
        html.Div(id="time_series_area_total"),
        html.Div(dcc.RangeSlider(
         id='total_range_slider',
         min=2000,
         max=2019,
         step=1,
         value=[2000, 2019],
        marks={
            2000: '2000',
            2001: '',
            2002: '2002',
            2003: '',
            2004: '2004',
            2005: '',
            2006: '2006',
            2007: '',
            2008: '2008',
            2009: '',
            2010: '2010',
            2011: '',
            2012: '2012',
            2013: '',
            2014: '2014',
            2015: '',
            2016: '2016',
            2017: '',
            2018: '2018',
            2019: ''
            }
        ), className="area_chart_range_slider")
    ], className="visualization_item_10 area_chart_total"),
    html.Div(["Select fuel type, year range and countries to view time series line chart"], className="visualization_item_11 line_chart_title"),
        html.Div(
            dcc.Dropdown(
                id='countries_dropdown_line',
            options=[
                {'label': i, 'value': i} for i in all_countries
            ],
             multi=True,
            value=["India", "Canada", "China", "South Korea"]
            )
            ,className= "visualization_item_12 country_dropdown_content_line"),
        html.Div(dcc.RadioItems(
    id="fossil_fuel_type_2",
    options=[
            {'label': ' Gas', 'value': 'gas'},
            {'label': ' Coal', 'value': 'coal'},
            {'label': ' Oil', 'value': 'oil'}
    ],
    value='gas',
     labelStyle={'display': 'inline-block',
                 'color':'#003399',
                 'padding':'0.27em 0 0 1em',
                 'font-size':'1.3em'}
    ), className="visualization_item_13"),
        html.Div(id="country_line_chart", className="visualization_item_14"
        ),
        html.Div(dcc.RangeSlider(
         id='total_range_line_slider',
         min=2000,
         max=2019,
         step=1,
         value=[2000, 2019],
        marks={
            2000: '2000',
            2001: '2001',
            2002: '2002',
            2003: '2003',
            2004: '2004',
            2005: '2005',
            2006: '2006',
            2007: '2007',
            2008: '2008',
            2009: '2009',
            2010: '2010',
            2011: '2011',
            2012: '2012',
            2013: '2013',
            2014: '2014',
            2015: '2015',
            2016: '2016',
            2017: '2017',
            2018: '2018',
            2019: '2019'
            }
        ), className="visualization_item_15 range_slider_line")
        ], className="climate_co2_visualizations")
        ]
    elif value == "temp":
        content = [
            html.Div([
                html.Div(["World temperature change line chart"],className="temp_item_1 world_line_title"),
                html.Div(dcc.Graph(figure=temp_world),className="temp_item_2"),
                html.Div("Select countries to view time series line chart", className="temp_item_3"),
                html.Div(dcc.Dropdown(
                id='temp_countries_dropdown_line',
            options=[
                {'label': i, 'value': i} for i in all_temp_countries
            ],
             multi=True,
            value=["India", "Canada", "China", "South Korea"],
            style={
                "width":"35em"
            }
            ), className="temp_item_4"),
            html.Div(id="temp_countries_line_chart",className="temp_item_5"),
            html.Div("Yearly temperature change geochart", className="temp_item_6"),
            html.Div(id="temp_change_geoplot",className="temp_item_7"),
            html.Div([
                    dcc.Slider(
                        id="temp_change_years",
                        min=2000,
                        max=2019,
                        marks={
                            2000: '2000',
                            2001: '2001',
                            2002: '2002',
                            2003: '2003',
                            2004: '2004',
                            2005: '2005',
                            2006: '2006',
                            2007: '2007',
                            2008: '2008',
                            2009: '2009',
                            2010: '2010',
                            2011: '2011',
                            2012: '2012',
                            2013: '2013',
                            2014: '2014',
                            2015: '2015',
                            2016: '2016',
                            2017: '2017',
                            2018: '2018',
                            2019: '2019'
                        },
                        value=2000,
                        included=False,
                        className="temp_slider"
                    )
                ], className="temp_item_8"),
                html.Div(["Select country to view violin plot"], className="temp_item_9"),
                html.Div(dcc.Dropdown(
                id='temp_countries_dropdown_violin',
            options=[
                {'label': i, 'value': i} for i in all_temp_countries
            ],
             multi=True,
            value=["India", "Canada", "China", "South Korea"]
            ), className="temp_item_10"),
            html.Div(id="temp_change_violin",className="temp_item_11")
            ], className="climate_temp_visualizations")
        ]
    return html.Div(content)


@app.callback(
    Output('exp_co2_data_description', 'children'),
    Input('dataset_fuel_type', 'value'),
)
def exp_desc(value):
    decs = data_description(value)
    if value == "coal":
        content = html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in decs.columns],
                data=decs.to_dict('records'),
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    },
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    }
                ],
                style_cell={'padding': '5px'},
                style_cell_conditional=[
                    {'if': {'column_id': 'Country'},
                     'width': '6em'},
                    {'if': {'column_id': 'Pts'},
                     'width': '4em'},
                ],
                style_data={
                    'font-family': '"Verlag", sans-serif',
                    'text-align': 'center',
                    'font-size': '12px',
                    'padding': '6px',
                },
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'width': '71em'
                },
                style_header={
                    'padding': '10px',
                    'font-family': '"Verlag", sans-serif',
                    'backgroundColor': '#002D72',
                    'color': 'white',
                    'text-align': 'center',
                    'font-size': '15px'
                }
            )
        ])
    elif value == "oil":
        content = html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in decs.columns],
                data=decs.to_dict('records'),
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    },
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    }
                ],
                style_cell={'padding': '5px'},
                style_cell_conditional=[
                    {'if': {'column_id': 'Country'},
                     'width': '6em'},
                    {'if': {'column_id': 'Pts'},
                     'width': '4em'},
                ],
                style_data={
                    'font-family': '"Verlag", sans-serif',
                    'text-align': 'center',
                    'font-size': '12px',
                    'padding': '6px',
                },
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'width': '71em'
                },
                style_header={
                    'font-family': '"Verlag", sans-serif',
                    'backgroundColor': '#002D72',
                    'color': 'white',
                    'text-align': 'center',
                    'font-size': '15px'
                }
            )
        ])
    else:
        content = html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in decs.columns],
                data=decs.to_dict('records'),
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    },
                    {
                        'if': {'row_index': 'even'},
                        'backgroundColor': '#1877F2',
                        'color': 'white'
                    }
                ],
                style_cell={'padding': '5px'},
                style_cell_conditional=[
                    {'if': {'column_id': 'Country'},
                     'width': '6em'},
                    {'if': {'column_id': 'Pts'},
                     'width': '4em'},
                ],
                style_data={
                    'font-family': '"Verlag", sans-serif',
                    'text-align': 'center',
                    'font-size': '12px',
                    'padding': '6px',
                },
                style_table={
                    'overflowX': 'auto',
                    'overflowY': 'auto',
                    'width': '71em'
                },
                style_header={
                    'padding': '10px',
                    'font-family': '"Verlag", sans-serif',
                    'backgroundColor': '#002D72',
                    'color': 'white',
                    'text-align': 'center',
                    'font-size': '15px'
                }
            )
        ])
    return html.Div(content)

@app.callback(
    Output('count_plot', 'children'),
    Output('count_plot_1', 'children'),
    Output('count_plot_2', 'children'),
    Output('count_plot_3', 'children'),
    Input('dataset_fuel_type_1', 'value'),
)
def exp_desc(value):
    coun, ran, to, tol = exp_vis(value)
    return html.Div(dcc.Graph(figure=coun)), \
           html.Div(dcc.Graph(figure=ran)), \
           html.Div(dcc.Graph(figure=to)), \
           html.Div(dcc.Graph(figure=tol))

@app.callback(
    Output('exp_violin_plot', 'children'),
    Input('dataset_fuel_type_1', 'value'),
    Input('countries_dropdown_2', 'value'),
)
def exp_violin(type, countries):
    fig = exp_country_violin(countries, type)
    return html.Div(dcc.Graph(figure=fig))

@app.callback(
    Output('exp_temp_top_line', 'children'),
    Input('year_dropdown_temp', 'value'),
)
def temp_top(year):
    fig = top_temp_bar(year)
    return html.Div(dcc.Graph(figure=fig))

@app.callback(
    Output('stat_models', 'children'),
    Input('stats_radio_button', 'value'),
)
def stat_models(value):
    elbow, cluster = cluster_analysis()
    train_fig, fore_fig = arima_model()
    train_fig.update_layout(margin={"r":0,"t":0,"l":0,"b": 0}, height= 420)
    fore_fig.update_layout(margin={"r":0,"t":0,"l":0,"b": 0}, height= 420)
    temp_world = world_temp()
    if value == "cluster":
        content = html.Div(
            [
                html.Div([
                html.Div("K Means Clustering and determining number of clusters", className="cluster_item_1 cluster_item_title"),
                html.Div("K-means clustering algorithm tries to group similar items in the form of clusters."
                         " The number of groups is represented by K."
                         " It finds the similarity between the items and groups them into the clusters."
                         " Here we will be clustering countries into similar group based on Co2 emission levels for coal, oil, and gas fossil fuel type for the past 20 years"
                         " In order to determine the value of K we will be using the Elbow method"
                         " It is an empirical method to find out the best value of k."
                         " It picks up the range of values and takes the best among them."
                         " It calculates the sum of the square of the points and calculates the average distance."
                         " From the elbow plot we can see that the best value for K is 5.", className="cluster_item_4"),
                html.Div(dcc.Graph(figure=elbow), className="cluster_item_5"),
                html.Div("Model Implemented", className="cluster_item_6 cluster_item_title"),
                html.Div("Before we implement the k-means model we need to scale the data."
                         " The main reason for scaling is so that the algorithm should not be biased towards variables which may have a higher magnitude."
                         " If you don’t normalize your features, you will end up giving more weight to some features than others."
                         " In order to overcome this problem, we need to bring down all the variables to the same scale."
                         " We will be using min-max scaling to get all the variable to the same scale."
                         " After the data was brought down to the same scale the k-means algorithm was implemented by setting k as 5."
                         " The scatter plot on the right shows the coutries that were group in the similar cluster.", className="cluster_item_7"),
                html.Div(dcc.Graph(figure=cluster), className="cluster_item_8"),
                html.Div("Countries present in each cluster", className="cluster_item_9 cluster_item_title"),
                html.Div(html.Img(src='assets/Cluster.jpg', className="cluster_image"), className="cluster_item_10"),
                ],className="cluster_model_content")
            ])
    elif value == "forecast":
        content = html.Div(
            [
                html.Div([
                html.Div("ARIMA model and checking if data is stationary", className="forecast_item_1 forecast_item_title"),
                html.Div("An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time series data to either better understand the data set or to predict future trends."
                         " Autoregressive integrated moving average (ARIMA) models predict future values based on past values."
                         " In order to implement the arima model we need to check if the data is stationary or not for that we will be using the ADF (Augmented Dickey-Fuller) test."
                         " From the ADF test we get a p-value of 6.792965211847253e-07 which means that the we can reject the null hypothesis."
                         " Therefore, infering that the data is stationary", className="forecast_item_2"),
                html.Div(dcc.Graph(figure=temp_world), className="forecast_item_3"),
                html.Div("Best Model for data", className="forecast_item_4 forecast_item_title"),
                html.Div("For an ARIMA model we need to provide the model with p, d and q values."
                         " We will be using the iterative approach to find the optimal p, d and q values that will be best suited for this dataset."
                         " We can see that the model with the values (9, 1, 0) is best suited for this dataset.", className="forecast_item_5"),
                html.Div("Dividing into testing set, training set and validating the model", className="forecast_item_6 forecast_item_title"),
                html.Div("We will be dividing the dataset into training and testing set where that last 9 instance will be to test the model."
                         " We will be using the RSME score to validate the model which basically tells us the difference between the predicted and observed values."
                         " We get a RSME score of 0.187 for the model with values (9, 1, 0).", className="forecast_item_7"),
                html.Div(dcc.Graph(figure=train_fig), className="forecast_item_8"),
                html.Div("Predicting future values", className="forecast_item_9 forecast_item_title"),
                html.Div("We then used this model to predict the next 10 years of temperature change values", className="forecast_item_10"),
                html.Div(dcc.Graph(figure=fore_fig), className="forecast_item_11"),

                ],className="forecast_model_content")
            ])
    else:
        content = "Error"
    return html.Div(content)

@app.callback(
    Output("temp_change_violin", "children"),
    Input("temp_countries_dropdown_violin", "value")
)
def violin(temp_countries):
    fig = temp_country_violin(temp_countries)
    return html.Div(dcc.Graph(figure=fig))

@app.callback(
    Output("temp_countries_line_chart", "children"),
    Input("temp_countries_dropdown_line", "value")
)
def line_chart(temp_country):
    fig = temp_country_line(temp_country)
    return html.Div(dcc.Graph(figure=fig))

@app.callback(
    Output("country_line_chart", "children"),
    Input("total_range_line_slider", "value"),
    Input("countries_dropdown_line", "value"),
    Input("fossil_fuel_type_2", "value")
)
def line_chart(ran, country, fuel_type):
    fig = line_chart_countries(ran, country, fuel_type)
    return html.Div(dcc.Graph(figure=fig))

@app.callback(
    Output("time_series_area_total", "children"),
    Input("total_range_slider", "value")
)
def time_series_area(range):
    graph = time_series_area_graph(range)
    return html.Div(dcc.Graph(figure=graph), className="area_plot")

@app.callback(
    dash.dependencies.Output('climate_gas_value', 'children'),
    [dash.dependencies.Input('countries_dropdown', 'value'),
     dash.dependencies.Input('year_dropdown', 'value'),
     ])
def update_output(country, year):
    climate_data = fossil_fuel("gas")
    gas = climate_data[["Countries", year]]
    gas_value = gas[gas["Countries"] == country].values[0, 1]
    return [html.Div([
            html.Img(src='assets/gas.png', style={'width': '5.5em', 'height': '5.5em'}, className="image_item"),
            html.H2("MtCO2 Gas", className="number_gas_title title_box"),
            html.H1(gas_value, className="number_gas_value number_box")
        ], className="number_box_content")
    ]

@app.callback(
    dash.dependencies.Output('climate_coal_value', 'children'),
    [dash.dependencies.Input('countries_dropdown', 'value'),
     dash.dependencies.Input('year_dropdown', 'value'),
     ])
def update_output(country, year):
    climate_data = fossil_fuel("coal")
    coal = climate_data[["Countries", year]]
    coal_value = coal[coal["Countries"] == country].values[0, 1]
    return [html.Div([
            html.Img(src='assets/coal.png', style={'width': '5.5em', 'height': '5.5em'}, className="image_item"),
            html.H2("MtCO2 Coal", className="number_coal_title title_box"),
            html.H1(coal_value, className="number_coal_value number_box")
        ], className="number_box_content")
    ]

@app.callback(
    dash.dependencies.Output('climate_oil_value', 'children'),
    [dash.dependencies.Input('countries_dropdown', 'value'),
     dash.dependencies.Input('year_dropdown', 'value'),
     ])
def update_output(country, year):
    climate_data = fossil_fuel("oil")
    oil = climate_data[["Countries", year]]
    oil_value = oil[oil["Countries"] == country].values[0, 1]
    return [html.Div([
            html.Img(src='assets/oil.png', style={'width': '5.5em', 'height': '5.5em'}, className="image_item"),
            html.H2("MtCO2 Oil", className="number_oil_title title_box"),
            html.H1(oil_value, className="number_oil_value number_box")
        ], className="number_box_content")
    ]

@app.callback(
    dash.dependencies.Output('geo_plot_output', 'children'),
    [dash.dependencies.Input('fossil_fuel_type', 'value'),
     dash.dependencies.Input('fossil_fuels_year', 'value'),
     ])
def geo_data(type, year):
    climate_data = fossil_fuel(type)
    choropleth = climate_change_choropleth(climate_data, year)
    return html.Div(dcc.Graph(figure=choropleth))

@app.callback(
    dash.dependencies.Output('polar_countries', 'children'),
    [dash.dependencies.Input('polar_countries_dropdown', 'value'),
     ]
)
def radar_plot(countries):
    data = join_fossil()
    radar = fossil_fuel_radar(data, countries)
    return html.Div(dcc.Graph(figure=radar))

@app.callback(
    Output("temp_change_geoplot", "children"),
    Input("temp_change_years", "value")
)
def temp_geoplot(year):
    world_temp = world_temp_geo(year)
    return html.Div(dcc.Graph(figure=world_temp))

if __name__ == '__main__':
    # Beanstalk expects it to be running on 8080.
    application.run(debug=True, port=8080)