import plotly.graph_objects as px

tag = ['Positive', 'Negative']
actual_sentiment = [5, 5]
predicted_sentiment = [6, 4]

plot = px.Figure(data=[px.Bar(
    name='Actual Sentiment',
    x=tag,
    y=actual_sentiment
),
    px.Bar(
        name='Predicted Sentiment',
        x=tag,
        y=predicted_sentiment
    )
])

plot.update_layout(
    width=800,
    height=600
)

plot.show()
