import pandas as pd
import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.figure_factory as px
import plotly.figure_factory as line
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.title("Esatisticas da Saúde Mental no Período de 2004 a 2020 ")

st.write(
    """Estudo Efetuado pelo INE - Instiituto Nacional de Estatística sobre o estado da Saúde Mental dos Portugueses 
    divídidos por Género, Faixa Étaria e Ocupação Profissional
    """
)
def filter_data(df: pd.DataFrame) ->pd.DataFrame:
    options = st.multiselect("escolha a Cena ", options=df.columns)
    st.write('Voçê selecionou as seguintes opções', options)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """

    modify = st.text_input(
        "Escolha os Fatores 👇", df.columns,
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,

    )
    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

def filter_dataframe2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify2 = st.text_input(
        "Escolha os Fatores 👇",
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,
    )

    if not modify2:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Selecione os Riscos", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

#df = pd.read_csv(
    #"MentalHealth.csv"
#)
#st.dataframe(filter_dataframe(df))

df = pd.read_csv(
    "Mentalhealth3.csv"
)

st.dataframe(filter_dataframe2(df))

chart_data = pd.DataFrame(
    np.random.randn( 22 , 5),
    columns=['Mulheres', 'Homens', 'Ensino Superior', 'Desempregados', 'Reformados' ])


    #column2=['25','50','75', '80', '100']
st.area_chart(chart_data)

# Example dataframe
df = pd.read_csv('Mentalhealth3.csv')

# plot
st.area_chart(data = df, x= "Date1",y='Total')



chart_data = pd.DataFrame(
    np.random.randn( 2 , 5),
    {

    'Date1': [2004,2008,2010,2015,2022],
    'columns':['Mulheres', 'Homens', 'Ensino Superior', 'Desempregados', 'Reformados' ] })


    #column2=['25','50','75', '80', '100']
st.area_chart(chart_data)

df = pd.read_csv('Mentalhealth3.csv')
st.area_chart( df, x="Date1", y='Total')

df = pd.DataFrame(
    {"Date1": [2008, 2011, 2018, 2020], "values": [0, 25, 50, 75], "values_2": [15, 25, 45, 85]}

).set_index("Date1")

df_new = pd.DataFrame(
    {"steps": [4, 5, 6], "Homens": [0.5, 0.3, 0.5], "Mulheres": [0.8, 0.5, 0.3]}
).set_index("steps")

df_all = pd.concat([df, df_new], axis=0)
st.line_chart(chart_data, x=df.all,)
#st.line_chart(df, x=df.index, y=["Homens", "Mulheres"])







# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2

# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Homens', 'Mulheres', 'Ensino superior']

# Create distplot with custom bin_size
fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[2008, 2010, 2020])

# Plot!
st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import streamlit.components.v1 as components
p = open("lda.html")
components.html(p.read(), width=1000, height=800, )

