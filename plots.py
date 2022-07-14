#!/usr/bin/env python
# coding: utf-8

# In[25]:


import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pmdarima as pm 
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Different Classifiers")

st.write("""
# Explore different clssifier and datasets
Which one is best?
""")

dataset_name = st.sidebar.selectbox(
                'select Dataset',
                ('Iris','glass','car','Click-Through Rate'))

st.write(f"##{dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
                    'Select classifier',
                    ('KNN','SVM','Random Forest','Logistic Regression'))

chart_select = st.sidebar.selectbox(
                label = "select the chart type",
                options = ['Scatterplots', 'Countplots','Barplot'])


# In[32]:


def get_dataset(name):
    if name=='Iris':
        data=pd.read_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\Iris.csv")
        
        numeric_columns = list(data.select_dtypes(['float','int']).columns)

        if chart_select == 'Scatterplots':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options= numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options= numeric_columns)
                plot = px.scatter(data_frame=data, x= x_values, y = y_values,color="Species")
                # display the chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
                
        if chart_select == 'Countplots':
            st.sidebar.subheader('Countplots Settings')
            sns.countplot(data["Species"])
            st.pyplot()

        
        x = data.drop(['Species'],axis=1)
        y= data["Species"]
    
    elif name== 'glass':
        data= pd.read_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\glass.csv")
        
        numeric_columns = list(data.select_dtypes(['float','int']).columns)

        if chart_select == 'Scatterplots':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options= numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options= numeric_columns)
                plot = px.scatter(data_frame=data, x= x_values, y = y_values,color="Type")
                # display the chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
                
        elif chart_select == 'Countplots':
            st.sidebar.subheader('Countplots Settings')
            sns.countplot(data["Type"])
            st.pyplot()
         
        elif chart_select == 'Barplot':
            st.sidebar.subheader('Barplot Settings')
            try:
                x_columns = list(data.select_dtypes(['float','int']).columns)
                fig = plt.figure()
                sns.barplot(x=data['Type'], y = st.sidebar.selectbox('X axis', options= x_columns),data=data)
                st.pyplot(fig)
            except Exception as e:
                print(e)
        
        x= data.drop(['Type'],axis=1)
        y= data['Type']
    
    elif name== 'car':
        data= pd.read_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\car.csv")
        data.columns=['buying','maint','doors','persons','lug_boot','safety','class']
        
        numeric_columns = list(data.select_dtypes(['object']).columns)

        if chart_select == 'Countplots':
            st.sidebar.subheader('Countplots Settings')
            fig = plt.figure()
            sns.countplot(data[st.sidebar.selectbox('X axis', options= numeric_columns)])
            st.pyplot()   
        
        
        le=LabelEncoder()
        for x in data:
            data[x]=le.fit_transform(data[x])

        x=data.drop(['class'],axis=1)
        y= data['class']

        scaler=StandardScaler()
        scaler.fit(x)
        x=scaler.transform(x)
        
    if name=='Click-Through Rate':
        data=pd.read_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\ad.csv")
        
        numeric_columns = list(data.select_dtypes(['float','int']).columns)

        if chart_select == 'Scatterplots':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options= numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options= numeric_columns)
                plot = px.scatter(data_frame=data, x= x_values, y = y_values,color="Clicked on Ad")
                # display the chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
                
        if chart_select == 'Countplots':
            st.sidebar.subheader('Countplots Settings')
            sns.countplot(data["Clicked on Ad"])
            st.pyplot()
            
        elif chart_select == 'Barplot':
            st.sidebar.subheader('Barplot Settings')
            try:
                x_columns = list(data.select_dtypes(['float','int']).columns)
                fig = plt.figure()
                sns.barplot(x=data['Clicked on Ad'], y = st.sidebar.selectbox('X axis', options= x_columns),data=data)
                st.pyplot(fig)
            except Exception as e:
                print(e)

        
        le=LabelEncoder()
        for x in data:
            data[x]=le.fit_transform(data[x])

        data=data.drop(['Ad Topic Line','City'],axis=1)
        x = data.drop(['Clicked on Ad'],axis=1)
        y= data["Clicked on Ad"]
        
    if name=='Bitcoin':
        #web scraping using selenium
#         from selenium import webdriver
#         driver=webdriver.Chrome()
#         driver.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/')
#         tr1=[driver.find_element_by_xpath('//*[@id="__next"]/div[1]/div[1]/div[2]/div/div[3]/div/div/div[1]/div[2]/table')]
#         data=tr1[0].text
#         print(data)
#         import pandas as pd
#         data = pd.DataFrame([x.split('$') for x in data.split('\n')],columns=['Date','open','High','Low','close','Volume','Market_cap'])
#         df.drop([0], axis=0, inplace=True)
#         df.to_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\BTC_2018-2022.csv",index=0)
        
        data=pd.read_csv(r"C:\Users\USER\Desktop\imarticus\Caspton_project\BTC_2018-2022.csv")
        data['Date']=pd.to_datetime(data['Date']) # to convert object to datetime
        close_price=data[['Date','close']]  # Taking Date and Closing price of BTC
        close_price=close_price.set_index("Date")  # Makeing Date as Index number
        y=close_price['close'].resample("d").mean()  # Doing perdiction on Days  
        
        numeric_columns = list(data.select_dtypes(['float','int']).columns)

        if chart_select == 'Scatterplots':
            st.sidebar.subheader('Scatterplot Settings')
            try:
                x_values = st.sidebar.selectbox('X axis', options= numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options= numeric_columns)
                plot = px.scatter(data_frame=data, x= x_values, y = y_values,color="Species")
                # display the chart
                st.plotly_chart(plot)
            except Exception as e:
                print(e)
        if chart_select == 'Countplots':
            st.sidebar.subheader('Countplots Settings')
            sns.countplot(data["Species"])
            st.pyplot()

        train=y.loc[:'2020-12-31']

               
    if name =='Bitcoin':
        return y
    else:
        return x, y     


# In[33]:

if dataset_name =='Bitcoin':
    y = get_dataset(dataset_name)
    st.write('shape of dataset', y.shape)
else:
    x, y = get_dataset(dataset_name)
    st.write('shape of dataset', x.shape)
    st.write('Number of class', len(np.unique(y)))


# In[34]:


def add_parameter(classifier):
    params=dict()
    if classifier=='SVM':
        C=st.sidebar.slider('C', 0.01,10.0)
        gamma=st.sidebar.slider('gamma', 0.1,0.001)
        params['C']=C
        params['gamma']=gamma
    elif classifier == 'KNN':
        K=st.sidebar.slider('K',1,10)
        params['K']=K
    elif classifier == "Random Forest":
        N=st.sidebar.slider('N', 10,100)
        params['N']=N
        
    elif classifier == "Logistic Regression":
        C=st.sidebar.slider('c', 0.001,2.0)
        params['C']=C
        
    elif classifier == "Time Series":
        p=st.sidebar.slider('p',0,10)
        q=st.sidebar.slider('d',0,10)
        d=st.sidebar.slider('q',0,10)
        P=st.sidebar.slider('P',0,10)
        Q=st.sidebar.slider('D',0,10)
        D=st.sidebar.slider('Q',0,10)
        m=st.sidebar.slider('S',1,12)
        params['p']=p
        params['d']=d
        params['q']=q
        params['P']=P
        params['D']=D
        params['Q']=Q
        params['m']=m
    
    return params


# In[35]:


params= add_parameter(classifier_name)


# In[36]:


def get_classifier(classifier,params):
    clf=None
    if classifier=='SVM':
        clf=SVC(C=params['C'],gamma=params['gamma'])
    elif classifier=='KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier=='Random Forest':
        clf=RandomForestClassifier(n_estimators=params['N'])
        
    elif classifier=='Logistic Regression':
            clf=LogisticRegression(C=params['C'])
        
    try:
        if classifier=='Time Series':
            clf=SARIMAX(order=params['p','d','q'],seasonal_order=params['P','D','Q','m']).fit()
    except KeyError:
        return clf
    return clf


# In[37]:


clf=get_classifier(classifier_name,params)


# In[38]:


#### CLASSIFICATION ####
if dataset_name == 'Bitcoin':
    train=y.loc[:'2020-12-31']
    Y_test=y.loc['2021-01-01':]
    try:
        train.apply(clf.fit())
    except Exception as e:
        print(e)        
    #clf.fit(train)
    try:
        y_pred = clf.predict(test)

        
        #print("RMSE value:",rms)
    except Exception as e:
        print(e)
    
    pred = clf.get_forecast(steps=529)
    Y_pred=pred.predicted_mean    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    rms=sqrt(mean_squared_error(Y_test,Y_pred))
    st.write(f'classifier = {classifier_name}')
    st.write(f'Accuracy =', rms)
    


else:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)


    st.write(f'classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)



# In[39]:


###### PLot Dateset #######


# In[40]:

def scatterplot(dataset_name):
    if dataset_name != 'Bitcoin':
        pca = PCA(2)
        x_projected= pca.fit_transform(x)
        abc=pd.DataFrame(x_projected,columns=['PCA1','PCA2'])
        abc['y']=y
        sns.scatterplot(data=abc, x="PCA1", y="PCA2", hue="y")



        plt.show()
        st.pyplot()


# In[41]:


scatterplot(dataset_name)


# In[ ]:




