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
import csv
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Estatísticas da Saúde Mental no Período de 2004 a 2020 ")

st.write(
    """Estudo Efetuado pelo INE - Instiituto Nacional de Estatística sobre o estado da Saúde Mental dos Portugueses 
    divídidos por Género, Faixa Étaria e Ocupação Profissional
    """
)


with st.sidebar:
    with st.sidebar:
        st.title(" WHY AI FOR HEALTH CARE?")
        st.write(""" 1. Organize and analyze data: A virtual assistant can help organize and analyze large amounts of data related to mental health, such as patient demographics, treatment outcomes, and resource utilization. 2. This can provide valuable insights for healthcare providers and policymakers to improve mental health services.

3. Identify patterns and trends: By using data analysis tools, a virtual assistant can identify patterns and trends in mental health data. 4. This can help identify high-risk populations, common mental health conditions, and areas where resources are lacking.

5. Predictive modeling: With the help of machine learning algorithms, a virtual assistant can create predictive models to forecast future mental health needs and resource requirements. This can assist in planning and allocating resources effectively.

6. Improve patient care: By analyzing patient data, a virtual assistant can identify gaps in care and suggest interventions to improve patient outcomes. This can include personalized treatment plans, medication adherence reminders, and follow-up care recommendations.

7.Monitor mental health trends: A virtual assistant can continuously monitor mental health data and alert healthcare providers to any significant changes or emerging trends. This can help in early detection and prevention of mental health issues.

8. Provide resources and support: A virtual assistant can act as a virtual resource center, providing information and support to individuals seeking mental health resources. """)
        st.title("Pode Adicionar outro daTa Set em CSV")
        st.write("Apenas Necessita de Adicionar um novo CSV")
        Button = st.button("Adicionar outro CSV")  
        if Button == True:
            File = st.file_uploader("Adcione aqui dados sobre saúde", type={"csv"})
            try:
                if File is not None:
                    df = pd.read_csv(File, low_memory=False)
            except valueError:
                print("Não Foi Adicionado CSV")

def filter_data(df: pd.DataFrame) ->pd.DataFrame:
    options = st.multiselect("escolha a Cena ", options=df.columns)
    st.write('Voçê selecionou as seguintes opções', options)
    #adicionei aqui uma cena nova
    df = pd.read_csv('Mentalhealth3.csv')
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.multiselect(
    'Fatores de Risco👇',
    ['Obsidade', 'Diabetes', 'Risco de Saúde', 'Tensão Alta',
    'Outras Morbilidades', 'Antecedentes familiares'])

       # "Escolha os Fatores 👇", df.columns,
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,
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
    modify2 = st.multiselect(
        'Fatores de Risco👇',
    ['Obsidade', 'Diabetes', 'Risco de Saúde', 'Tensão Alta',
    'Outras Morbilidades', 'Antecedentes familiares'])
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #placeholder=st.session_state.placeholder,
    

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
    
    

#End
df = pd.read_csv(
    "MentalHealth3.csv"
)
st.dataframe(filter_dataframe(df))
st.write("____________________________________________________________") 


#df = pd.read_csv(
    #"Mentalhealth3.csv"
#)
#######inicio dAS TABS
tab1, tab2, tab3, tab4 , tab5 = st.tabs(["The DataFrame","The Maximum Values", "The Minumum Values", "The Average Values", "Standard Deviation"])
with tab1:
    
    st.title("Data Science for Health") 
    
with tab2:
    st.header("The Maximum Values")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("")
        #st.write(" O resultado dos  dos Valores Máximos", df.max())
    with col2:
        
        df = pd.read_csv('Taxa de Desemprego 3.csv')
        Indx =  df.get('Date1')
        arr1  = df.get('Homens')
        arr2  = df.get('Mulheres')
        arr3  = df.get('Desempregados')
        arr4 =df.get('Ensino superior')
    
        marks_list = df['Date1'].tolist()
    
        marks_list2 = df['Desempregados'].tolist()
    
        marks_list5 = df['Homens'].tolist()
        marks_list3 = df['Mulheres'].tolist()
    
    
        marks_list4 = df['Ensino superior'].tolist()
    
        dict = {'Desempregados': marks_list2, 'Mulheres': marks_list3, 'Ensino superior': marks_list4, 'Homens' : marks_list5} 
        
        df1 = pd.DataFrame(dict)
        #st.write(max(df))
        chart_data = pd.DataFrame(df, columns=["Desempregados", "Mulheres", "Ensino superior", "Homens"])
    
        st.line_chart(chart_data)
with tab3:
    st.header("The Minumum Values")
    st.write(" O resultado dos  dos Valores minimos", df.min())
    
with tab4:
    st.header("The Average Values")
    col1, col2 = st.columns(2)
    with col1:
        st.write(" O resultado da média dos Valores é", df.mean())
    with col2:
        st.area_chart(data = df.mean())
       
with tab5:
    st.header("Standard Deviation")
    col1, col2 = st.columns(2)
    with col1:
        st.write(" O resultado da variancia", np.std(df))
    with col2:
        st.area_chart(data = np.std(df))

######FIM DAS TABS



st.dataframe(filter_dataframe2(df))
chart_data = pd.DataFrame(
np.random.randn( 22 , 5),
columns=['Mulheres', 'Homens', 'Ensino Superior', 'Desempregados', 'Reformados' ])


#column2=['25','50','75', '80', '100']
st.area_chart(chart_data)

#st.dataframe(filter_dataframe2(df))
    

st.write("____________________________________________________________")  
st.header("Valores Na relação Desempregados / Ensino Superior ")
# Example dataframe
df3 = pd.read_csv('Mentalhealth3.csv')

df_binary = df[['Desempregados', 'Ensino superior']]
 
# Taking only the selected two attributes from the dataset
df_binary.columns = ['Desempregados', 'Ensino superior']
#display the first 5 rows
df_binary.head()

marks_list = df3['Date1'].tolist()

marks_list2 = df3['Desempregados'].tolist()

marks_list5 = df3['Homens'].tolist()
marks_list3 = df3['Mulheres'].tolist()


marks_list4 = df3['Ensino superior'].tolist()

dict = {'Desempregados': marks_list2,  'Ensino superior': marks_list4,} 
    
df3 = pd.DataFrame(dict)
    
print(df3)

chart_data = pd.DataFrame(df, columns=["Desempregados",  "Ensino superior", ])

st.line_chart(chart_data)




#st.area_chart(data = df.mean())
st.write("____________________________________________________________") 
st.header("Relação com a Situação de Emprego")
#st.header("Evolução dos Valores Máximos")
#data = pd.DataFrame(np.random.randn(5, 3), columns=["2004", "2010", "2020"])

#st.line_chart(data= df.max())
df4 = pd.read_csv('Mentalhealth3.csv')
Indx =  df4.get('Date1')
arr1  = df4.get('Homens')
arr2  = df4.get('Mulheres')
arr3  = df4.get('Desempregados')
arr4 =df4.get('Ensino superior')

marks_list = df4['Date1'].tolist()

marks_list2 = df4['Desempregados'].tolist()

marks_list5 = df4['Homens'].tolist()
marks_list3 = df4['Mulheres'].tolist()


marks_list4 = df4['Ensino superior'].tolist()

dict = {'Desempregados': marks_list2, 'Mulheres': marks_list3, 'Ensino superior': marks_list4, 'Homens' : marks_list5} 
    
df4 = pd.DataFrame(dict)
    
print(df4)

chart_data = pd.DataFrame(df, columns=["Desempregados", "Mulheres", "Ensino superior", "Homens"])

st.line_chart(chart_data)

# Adding legend for stack plots is tricky.
#plt.plot(marks_list,marks_list2, color='b', label = 'Desempregados')
#plt.plot(marks_list,marks_list3, color='r', label = 'Mulheres')
#plt.plot(marks_list,marks_list4, color='y', label = 'Ensino superior')
#plt.plot( color='g', label = 'Desempregados')


#plt.stackplot( marks_list, marks_list2,  colors= ['r', 'g'])
#plt.title('Relação entee Mulheres no Desemprego com o ensino Superio')
#plt.legend()
#plt.show()



st.write("____________________________________________________________") 
st.title("Evolução dos Profissionais com Ensino Superior")
df5 = pd.read_csv('Mentalhealth3.csv')
st.area_chart( df5, x="Date1", y='Ensino superior')

df5 = pd.DataFrame(
    {"Date1": [2008, 2011, 2018, 2020], "values": [0, 25, 50, 75], "values_2": [15, 25, 45, 85]}

).set_index("Date1")

df_new = pd.DataFrame(
    {"steps": [2001, 2010, 2020], "Homens": [0.5, 0.3, 0.5], "Mulheres": [0.8, 0.5, 0.3]}
).set_index("steps")

df_all = pd.concat([df, df_new], axis=0)
st.line_chart(chart_data, x=df.all,)
st.write("_______________________________________________________")

df6 = pd.read_csv("Mentalhealth3.csv")

@st.experimental_memo

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


csv = convert_df(df6)

st.download_button(
   "Pode Fazer o Download dos Dados",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

st.write("_______________________________________________________")
st.write("Trabalho de Pesquisa e Programação: Paulo Ricardo Monteiro")
st.write("Formação em Fundamentos de Python Avançado por José Luis Boura - 2023/2024")
#st.line_chart(df, x=df.index, y=["Homens", "Mulheres"])

# Plot!
#st.plotly_chart(fig, use_container_width=True)

#import streamlit as st
#import streamlit.components.v1 as components
#p = open("lda.html")
#components.html(p.read(), width=1000, height=800, )

