# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from matplotlib import pyplot as plt
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from wordcloud import WordCloud, STOPWORDS

# Declaring Global variables
project_name = None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function definitions
def load_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv("scrappedReviews.csv")

    global pickle_model
    file = open("pickle_model.pkl", "rb")
    pickle_model = pickle.load(file)

    global vocab
    file = open("features.pkl", "rb")
    vocab = pickle.load(file)

    print("Loading ......")

    # Creating the pie chart
    temp = []
    for i in scrappedReviews["reviews"]:
        temp.append(check_review(i)[0])
    scrappedReviews["sentiment"] = temp
    positive = len(scrappedReviews[scrappedReviews["sentiment"] == 1])
    negative = len(scrappedReviews[scrappedReviews["sentiment"] == 0])
    explode = (0.1, 0)
    labels = [
        "Positive",
        "Negative",
    ]
    students = [positive, negative]
    colors = ["blue", "red"]
    plt.pie(
        students,
        explode=explode,
        startangle=90,
        colors=colors,
        labels=labels,
        autopct="%1.2f%%",
    )
    cwd = os.getcwd()
    if "assets" not in os.listdir(cwd):
        os.makedirs(cwd + "/assets")
    plt.savefig("assets/sentiment_chart.png")

    # Creating the wordcloud
    dataset = scrappedReviews["reviews"].to_list()
    str1 = ""
    for i in dataset:
        str1 = str1 + i
    str1 = str1.lower()
    stopwords = set(STOPWORDS)
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="yellow",
        stopwords=stopwords,
        min_font_size=10,
    ).generate(str1)
    cloud.to_file("assets/Word_Cloud.png")

    # Creating data to be put in dropdown
    global chart_dropdown_values
    chart_dropdown_values = {}
    for i in range(400, 501):
        chart_dropdown_values[scrappedReviews["reviews"][i]] = scrappedReviews[
            "reviews"
        ][i]
    chart_dropdown_values = [
        {"label": key, "value": values} for key, values in chart_dropdown_values.items()
    ]


def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocab)
    vectorised_review = transformer.fit_transform(
        loaded_vec.fit_transform([reviewText])
    )
    return pickle_model.predict(vectorised_review)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")


def create_app_ui():
    main_layout = html.Div(
        [
            html.H1(
                id="Main_title",
                children="Sentiment Analysis with Insights",
                style={
                    "text-align": "center", 
                    "color": "darkgreen"
                },
            ),
            html.Hr(style={"background-color": "lightgreen"}),
            html.H2(
                children="Pie Chart",
                style={
                    "text-align": "center",
                    "text-decoration": "underline",
                    "color": "brown",
                    "background-color": "lightgreen"
                },
            ),
            html.P(
                [
                    html.Img(
                        src=app.get_asset_url("sentiment_chart.png"),
                        style={
                            "width": "700px", 
                            "height": "400px"
                        },
                    )
                ],
                style={"text-align": "center"},
            ),
            html.Hr(style={"background-color": "lightgreen"}),
            html.H2(
                children="WordCloud",
                style={
                    "text-align": "center",
                    "text-decoration": "underline",
                    "color": "brown",
                    "background-color": "lightgreen"
                },
            ),
            html.P(
                [
                    html.Img(
                        src=app.get_asset_url("Word_Cloud.png"),
                        style={
                            "width": "700px", 
                            "height": "400px"
                        },
                    )
                ],
                style={"text-align": "center"},
            ),
            html.Hr(style={"background-color": "lightgreen"}),
            html.H2(
                children="Select a Review",
                style={
                    "text-align": "center",
                    "text-decoration": "underline",
                    "color": "blue",
                    "background-color": "lightgreen"
                },
            ),
            dcc.Dropdown(
                id="Chart_Dropdown",
                options=chart_dropdown_values,
                placeholder="Select a Review",
                style={
                    "font-size": "22px", 
                    "height": "70px", 
                    "color": "black", 
                    "background-color": "white"
                },
            ),
            html.H1(
                children="Missing",
                id="sentiment1",
                style={"text-align": "center", "color": "orange"},
            ),
            html.Hr(style={"background-color": "lightgreen"}),
            html.H2(
                children="Find Sentiment of Your Review",
                style={
                    "text-align": "center",
                    "text-decoration": "underline",
                    "color": "blue",
                    "background-color": "lightgreen"
                },
            ),
            dcc.Textarea(
                id="textarea_review",
                placeholder="Enter the review here.....",
                style={
                    "width": "100%", 
                    "height": 150, 
                    "font-size": "22px", 
                    "color": "black", 
                    "background-color": "white"
                },
            ),
            dbc.Button(
                children="Find Sentiment",
                id="button_review",
                style={
                    "width": "20%",
                    "background-color": "lightblue",
                    "color": "black", "font-weight": "bold"
                },
            ),
            html.H1(
                children="Missing",
                id="result",
                style={"text-align": "center", "color": "orange"},
            ),
        ]
    )

    return main_layout


@app.callback(
    Output("result", "children"),
    [Input("button_review", "n_clicks")],
    [State("textarea_review", "value")],
)
def update_app_ui(n_clicks, textarea_value):
    if n_clicks > 0:
        response = check_review(textarea_value)
        if response[0] == 0:
            result = "Negative"
        elif response[0] == 1:
            result = "Positive"
        else:
            result = "Unknown"
        return result
    else:
        return ""


@app.callback(Output("sentiment1", "children"), [Input("Chart_Dropdown", "value")])
def update_sentiment(review1):
    sentiment = []
    if review1:
        if check_review(review1) == 0:
            sentiment = "Negative"
        if check_review(review1) == 1:
            sentiment = "Positive"
    else:
        sentiment = "Missing"
    return sentiment


def main():
    print("Start of your project")
    load_model()
    open_browser()
    global scrappedReviews
    global project_name
    global app
    project_name = "Sentiment Analysis with Insights"
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    print("End of my project")
    project_name = None
    scrappedReviews = None
    app = None


# Calling the main function
if __name__ == "__main__":
    main()
