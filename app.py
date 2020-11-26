# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:14:59 2020

@author: thyn8
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

asof_date = '11/09/2020'
st.set_page_config(layout="wide")


def get_data_actual(fname):
    df = pd.read_excel(fname)
    df1 = df.groupby(['month'])['count'].sum().reset_index()
    df1['site'] = 'ALL SITES'
    df1 = df1.append(df)
    return df1

data_actual = get_data_actual('enrollment_actual.xlsx')


def get_data_plan(fname):
    df = pd.read_excel(fname)
    df1 = df.groupby(['month'])['plan_count'].sum().reset_index()
    df1['site'] = 'ALL SITES'
    df1 = df1.append(df)
    return df1

data_plan = (get_data_plan('enrollment_planned.xlsx'))


@st.cache
def get_start_end(df_actual,df_plan):
    
    df1=df_actual.sort_values(['site','month']).groupby('site').first().reset_index().drop('count',axis=1).rename(columns={'month':'date_start'})

    df2 = df_plan.copy()
    df2['total_planned'] = df2.groupby(['site'])['plan_count'].cumsum()
    df2 = df2.groupby(['site']).last().reset_index().drop('plan_count',axis=1).rename(columns={'month':'date_end'})

    df1=df1.merge(df2, on='site')
    return df1

def process_actual_plan(df_actual, df_plan, date_asof):
        
    df_goal = get_start_end(df_actual,df_plan)

    #Calculate total_enrolled for each site
    means = df_actual.groupby(['site'],as_index=False)['count'].sum().rename(columns={'count':'total_enrolled'})
    
    #Merge with dg_goal
    df_goal = df_goal.merge(means, on='site')
    
    #Create date_of variable
    df_goal['date_asof'] = date_asof
    df_goal['date_asof'] = df_goal['date_asof'].apply(pd.to_datetime, format='%m/%d/%Y')      
    
    #Convert string dates to datetime and calculate weeks past and remaining
    #df_goal[['date_start','date_end','date_asof']] = df_goal[['date_start','date_end','date_asof']].apply(pd.to_datetime,format='%Y-%m-/%d')  
    df_goal['weeks_left'] = (df_goal['date_end'] - df_goal['date_asof'])//np.timedelta64(1,'W') - 4
    df_goal['weeks_past'] = (df_goal['date_asof'] - df_goal['date_start'])//np.timedelta64(1,'W')

    #Calculate current and required rate
    df_goal['current_rate'] = df_goal['total_enrolled']/df_goal['weeks_past']
    df_goal['required_rate'] = (df_goal['total_planned'] - df_goal['total_enrolled'])/df_goal['weeks_left']
    
    #Merge with actual and planned data
    df_goal1 = df_goal.merge(df_actual, on='site')
    
    df_goal1 = pd.merge(df_plan,df_goal1,on=['site','month'],how='left')
    
    #Add cumulative sum for planned enrollment
    df_goal1['cumsum_plan'] = df_goal1.groupby(['site'])['plan_count'].cumsum()
    
    #Add cumulative sum of actual enrollment to merged data
    df_goal1['cumsum'] = df_goal1.groupby(['site'])['count'].cumsum()
    
    #Carry cumulative sum to end of study
    df_goal1['enrollment_stopped'] = df_goal1.groupby(['site'])['cumsum'].ffill()
    
    df_goal1['current_rate'] = df_goal1.groupby(['site'])['current_rate'].ffill()
    df_goal1['required_rate'] = df_goal1.groupby(['site'])['required_rate'].ffill()         
    
    # def project_cumsum(site):
    def project_cumsum(df, col, ratecol):
        df['index'] = df.index
        firstobs = df.groupby(['site'])['index'].first().tolist()
        for index, row in df.iterrows():
            if index not in firstobs and np.isnan(row[col]):
                previous = df[col][index-1]
                df.at[index,col] = previous+(4*row[ratecol]) #Add to cumulative sum
    
    #Calculate projected enrollment at current rate
    df_goal1['projected_current_rate'] = df_goal1['cumsum']
    project_cumsum(df_goal1, 'projected_current_rate', 'current_rate')
    
    df_goal1['projected_required_rate'] = df_goal1['cumsum']
    project_cumsum(df_goal1, 'projected_required_rate', 'required_rate')
            
    return [df_goal, df_goal1]


df_cumulative, df_detailed = process_actual_plan(data_actual,
                                                 data_plan,
                                                 date_asof = asof_date)



###################################################
#ENROLLMENT PROJECTIONS
###################################################
def fig_enroll_projections(df, choice, title):
    tdf = df[(df.site == choice)]
    #fig = px.Figure()
    fig = px.line(tdf,x='month',y=['cumsum_plan','projected_current_rate','projected_required_rate'], 
                  title=title,
                  line_shape='hv')
    #fig = fig.add_trace(px.line(temp_df,x='month',y='projected_required_rate', line_shape='hv'))
    #fig.add_trace(go.Scatter(x=temp_df['month'],y=temp_df['projected_required_rate'],name="Required Rate",line_shape='hv'))
    #fig.add_trace(go.Scatter(x=temp_df['month'],y=temp_df['projected_current_rate'],name="Current Rate",line_shape='hv'))
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01),
        autosize = True)
        #,
        #width = 1000,
        #height=500)

    return fig




###################################################
#PROCESS DATA FOR Difference in Rate Chart
###################################################
def data_dif_rate(df):
    tdf = df.copy()
    tdf['rate_diff'] = tdf['required_rate'] - tdf['current_rate']
    tdf = np.round(tdf,2)
    return tdf

def fig_dif_rate(df):
    fig = px.bar(df[df['site'] !='ALL SITES'], x='rate_diff',
                 y='site',
                 orientation= 'h',
                 hover_data=['current_rate','required_rate'])
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig


###################################################
#PROCESS DATA FOR Percent Achieve Chart
###################################################
def data_site_perc_achieve(df):
    tdf = df.copy()
    tdf['percent_enrolled'] = round((tdf['total_enrolled']/tdf['total_planned']) * 100,2)
    return tdf[['site','percent_enrolled','total_enrolled','total_planned']]

def fig_site_perc_achieve(df):
    fig = px.bar(df, x='site',
                 y='percent_enrolled',
                 hover_data=['site','total_enrolled','total_planned'])
    fig.update_layout(yaxis=dict(range=[0,100]))
    return fig
    

###################################################
#PROCESS DATA FOR GRANTT CHART
###################################################
def data_grant(df):           
    tdf = df.copy()
    #Add start time
    tdf['start'] = pd.to_datetime(tdf['month'])
    
    #Add finish time
    tdf['finish'] = tdf.groupby(['site'])['month'].shift(-1)
    tdf['finish'] = tdf['finish'].fillna(pd.to_datetime('2022-07-01'))

    #Drop observations after asof date
    #Fill in asof date
    tdf['date_asof'] = tdf['date_asof'].fillna(method='ffill').fillna(method='bfill')   
    tdf = tdf[tdf.month < tdf.date_asof]
    
    #Drop obervations with null cumulative sums of plan enrollment
    tdf = tdf[tdf['cumsum_plan'].notnull()]

    tdf['planned_enrollment'] = tdf['plan_count']
    tdf['actual_enrolled'] = tdf['count']
    
    tdf['complete'] = tdf['count']/tdf['plan_count']
    
    def flag_complete(df):
        if (df['plan_count'] == 0 and np.isnan(df['complete'])) or (df['complete'] >= 1):
            return '>=100%'
        elif (df['complete'] >=.75):
            return '75-99%'
        elif (df['complete'] >=.50):
            return '50-74%'
        else:
            return '0-49%'
    
    tdf['percent_complete'] = tdf.apply(flag_complete, axis = 1)
    tdf['month'] = pd.to_datetime(tdf['start'], format='%m').dt.month_name().str.slice(stop=3)
    tdf['year'] = pd.DatetimeIndex(tdf['start']).year 
    tdf['period'] = tdf['month'] + "-" + tdf['year'].astype(str)
        
    return tdf[['site','start','finish','period','actual_enrolled','planned_enrollment','percent_complete']]

# test = data_grant(df_detailed)

def fig_grant(df):
    fig = px.timeline(df, 
                      x_start="start", 
                      x_end="finish", 
                      y="site", 
                      color="percent_complete", 
                      range_x=[pd.to_datetime('2020-01-01'),pd.to_datetime('2023-01-01')],
                      hover_data= {'start':False,
                                   'finish':False,
                                   'site':True,
                                   'period':True,
                                   'actual_enrolled':True,
                                   'planned_enrollment':True,
                                   'percent_complete':True},
                      color_discrete_sequence=['#00CC96', '#AB63FA', '#FECB52','#EF553B'],
                      category_orders={'percent_complete':['>=100%','75-99%','50-74%','0-49%']})
    fig.update_yaxes(autorange="reversed")
    return fig


#####################################
#LAYOUT APP
#####################################

###################SIDE BAR
st.sidebar.write('CHAT 1901 Patient Enrollment')

###################TITLE BLOCK
st.markdown("<h1 style='text-align: center; color: black;'>CHAT 1901/1902 ENROLLMENT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>"+ "As of " + asof_date + "</h3>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


total_planned = 5000
total_enrolled = df_cumulative.loc[df_cumulative['site'] == 'ALL SITES']['total_enrolled'].values[0]
perc_enrolled = round(total_enrolled/total_planned*100,2)


st.markdown("<h2 style='text-align: center; color: black;'>Total Enrolled: " + str(total_enrolled) + "</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Total Planned: " +  str(total_planned) + "</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Percent Enrolled: " + str(perc_enrolled) + "%" + "</h2>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


###################ALL SITES BLOCK
#st.header('All Sites')
#cols = st.beta_columns((1,3,1))
#cols[1].plotly_chart(fig_enroll_projections(df_detailed, 'ALL SITES', title='All site projections'),use_container_width=True)


###################SITE SPECIFIC BLOC
st.markdown("<h2 style='text-align: center; color: black;'>Projected Enrollment</h2>", unsafe_allow_html=True)


cols = st.beta_columns((2,1,2))
site_list = df_cumulative['site'].tolist()
site= cols[1].selectbox(
       '',
      options=site_list)

cols = st.beta_columns((1,4,1))
cols[1].plotly_chart(fig_enroll_projections(df_detailed, site, title=''), use_container_width=True)


###################RATE DIFFERENCE
st.markdown("<h2 style='text-align: center; color: black;'>Difference between Current and Required Rate</h2>", unsafe_allow_html=True)

cols = st.beta_columns((1,2,1))
cols[1].plotly_chart(fig_dif_rate(data_dif_rate(df_cumulative)), use_container_width=True)

def fig_table(df):
    fig = ff.create_table(df)
    return fig

cols = st.beta_columns(3)
expander = cols[1].beta_expander("Toggle Data", expanded=False)
expander.plotly_chart(fig_table(data_dif_rate(df_cumulative)[['site','current_rate','required_rate','rate_diff']]),use_container_width=True)


# cols = st.beta_columns((2,1))
# cols[0].plotly_chart(fig_dif_rate(data_dif_rate(df_cumulative)), use_container_width=True)
# expander = cols[1].beta_expander("Toggle Data", expanded=False)
# expander.write()

#import plotly.io as pio
#pio.renderers.default = 'svg'
#pio.renderers.default = 'browser'

#PERCENT ENROLLED
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Percent Enrolled</h2>", unsafe_allow_html=True)

cols = st.beta_columns((1,4,1))
cols[1].plotly_chart(fig_site_perc_achieve(data_site_perc_achieve(df_cumulative)), use_container_width = True)

def data_totals(df):
    tdf = df.copy()
    tdf = tdf[['site','total_planned','total_enrolled']].T.reset_index()
    #tdf = tdf.style.hide_index()
#    tdf = tdf.rename(columns=dict(zip(tdf.columns.tolist(),list(tdf.loc['site'])))).drop('site')
    return tdf

my_expander1 = cols[1].beta_expander("Toggle Data", expanded=False)
my_expander1.write(data_totals(df_cumulative))

#GRANT CHART
cols[1].markdown("<h2 style='text-align: center; color: black;'>Monthly Percent Enrolled</h2>", unsafe_allow_html=True)

# cols = st.beta_columns(1)
cols[1].plotly_chart(fig_grant(data_grant(df_detailed)), use_container_width = True)

my_expander2 = cols[1].beta_expander("Toggle Data", expanded=False)
my_expander2.write(df_detailed.pivot(index='month',columns='site',values='count').reset_index())



