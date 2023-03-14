import time  # to simulate a real time data, time loop
import pandas as pd
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # data web app development
import pandas as pd
st.set_page_config(
    page_title="Real-Time - Data Science Dashboard",
    page_icon="",
    layout="wide",
)
import pandas as pd
df = pd.read_csv('MentalHealth.csv')
#df = pd.read_csv('https://query.data.world/s/edvqfu4ea2ap6sbmr7ht2szxjc2mij')
placeholder = st.empty()
#Title
st.title = "Real time computer Science Dashboard 1"

job_filter = st.selectbox("Select the Job", pd.unique(df["Sexo"]))

for seconds in range(200):
    df["Sexo"] = df["Sexo"] * np.random.choice(range(1, 5))
    df["Total"] = df["Total"] * np.random.choice(range(1, 5))

    # creating KPIs
    avg_age = np.mean(df["Total"])

    count_married = int(
        df[(df["Grupo etário"] == "Grupo etário")]["Grupo etário"].count()
        + np.random.choice(range(1, 1000))
    )

    balance = np.mean(df["Total"])

    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Sexo ⏳",
            value=round(avg_age),
            delta=round(avg_age) - 10,
        )

        kpi2.metric(
            label="Total 💍",
            value=int(count_married),
            delta=-10 + count_married,
        )

        kpi3.metric(
            label="Grupo etário",
            value=f"$ {round(balance, 2)} ",
            delta=-round(balance / count_married) * 100,
        )

import altair as alt
fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.title = "Location ID"
    st.markdown("Location ID")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Sexo', 'Total', 'Grupo etário'])

    c = alt.Chart(chart_data).mark_circle().encode(
        x='Sexo', y='Total', size='Grupo etário', color='Grupo etário', tooltip=['Sexo', 'Total', 'Grupo  etário'])

    st.altair_chart(c, use_container_width=True)



with fig_col2:
    st.markdown("Tree Map")
    fig = px.density_heatmap(
        data_frame=df, y="Sexo", x="Total"
    )
    st.write(fig)



st.title = "Road Map of Data Science"
st.markdown("road map")

df = pd.DataFrame(
np.random.randn(1000,2) / [50, 50] + [-86, 39],
columns=['lon', 'lat',])
st.map(df)

st.markdown("Detailed Data View")
st.dataframe(df)
