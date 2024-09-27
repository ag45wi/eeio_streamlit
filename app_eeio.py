import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px 
from io import StringIO

from util_app import plot_agg_each, plot_agg, save_toGit, save_toGit_xls, save_toGit_csv
from util_eeio import calc_mat

#chk pwd: /mount/src/eeio

#from _init_var import *
if 'IS_LOCAL' not in st.session_state:
    st.session_state['IS_LOCAL'] = 0  #1 for localhost; 0 for remote-host

if (st.session_state['IS_LOCAL']):
    print("st.session_state: LOCALHOST")
else:
    print("st.session_state: REMOTE HOST")

#https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
st.set_page_config(
    page_title="EEIO-Indonesia",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dashboard title
st.title('Indonesia Carbon Footprint Calculator')
st.subheader('Environment Sustainability Project')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Emission", "Data Center", "About"])

#print ("before tab1")
with tab1:
    # SIDEBAR
    #st.sidebar.header("This is my sidebar")

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader('Header column 1')

        fpath = "data/list_agg_sectors.csv"
        df_lagg = pd.read_csv(fpath)
        #print("df_lagg", df_lagg.head(5))
        lst = df_lagg['Aggregated sectors'].tolist()
        lst.insert(0, "--All Sectors--")

        opt_sector = st.selectbox("Category of Aggregated Sectors",lst)
        #print(type(opt_sector)) #str
    
    with col2:
        st.subheader('Header column 2')
        # Show static DataFrame
        #st.write(opt_sector)

        #opt_sector="Energy"
        if (opt_sector != "--All Sectors--"):
            df_agg_sectors_each = pd.read_csv("buf/result_agg_sectors_each.csv")
            df_selected=df_agg_sectors_each[df_agg_sectors_each["Aggregated sectors"] == opt_sector]
            
            plot_agg_each(df_selected, opt_sector)
            
        else:
            saved_file_path = "buf/result_agg_sectors.csv"
            df_agg_sectors = pd.read_csv(saved_file_path)
            df_agg_sectors.set_index('Aggregated sectors', inplace=True)
            #st.dataframe(df_agg_sectors)
            plot_agg(df_agg_sectors)

with tab2:
    col1, col2 = st.columns([2,3])
    with col1:
        f_io = st.file_uploader("Choose file: IO", type = 'csv')
        st.markdown("""---""")
        f_fec = st.file_uploader("Choose file: final energy consumption (FEC)", type = 'csv')
        st.markdown("""---""")
        f_conv = st.file_uploader("Choose file: conversion_factor", type = 'csv')
        st.markdown("""---""")
        f_co2 = st.file_uploader("Choose file: direct CO2 Emission Factor", type = 'csv')
        #f_agg = st.file_uploader("Choose aggregated_sectors (xlsx)", type = 'xlsx')

        DEF_name_io='io_ind_2016.csv'
        DEF_name_fec='final_energy_consumption_bytype.csv'
        DEF_name_conv='conversion_factor.csv'
        DEF_name_co2='direct_CO2_EF.csv'


        #button = st.button('Recompute Matrix', disabled=True)
        #print(type(button))


    with col2:
        st.write("**Existing or uploaded files**")
        def file_uploaded(fupload):
            #print(fupload.name, "is not None")
            st.write(f"Uploaded file: **{fupload.name}**")
            #st.dataframe(fupload)

            #df1=pd.read_excel(fupload)
            df1=pd.read_csv(fupload)
            #save_toGit(df1, fupload)
            #save_toGit_xls(df1, fupload.name)
            fname = fupload.name

            if (st.session_state['IS_LOCAL']):
                print("st.session_state.IS_LOCAL is True")
                df1.to_csv(f'data/{fname}', index=False)
            else:
                print("st.session_state.IS_LOCAL is False -- save to git")
                save_toGit_csv(df1, fname, "data")
            
            
            #df1=pd.read_excel(fupload)
            #try:
            #    df1.to_excel(fname, index=False)
            #except (e):
            #    print("except e:", e)
            df1_str = df1.astype(str) #just to avoid warning
            st.dataframe(df1_str)
            #st.write(",".join(str(x) for x in df1.columns.values))

            return fname
        
        def file_notUploaded(def_name):
            def_path=f"data/{def_name}"
            df1 = pd.read_csv(def_path)
            st.write(f"Existing file: **{def_path}**")
            col=df1.columns.values
            #st.write(x for x in col[:5])
            #print ("len col", len(col), col.shape)
            N=10
            if (len(col)<N):
                N=len(col)
            str1=""
            for i in range(N-1):
                curr_str=col[i]
                if (not curr_str): curr_str="[]"
                if (i==0): str1=curr_str
                else: str1=str1+", "+curr_str
                #print("str1 in loop:", str1)
            str1=str1+", " +col[N-1]
            if (len(col)>N): str1=str1+", ..."
            #print("str1:", str1)
            st.write("Fields: " + str1)
            #st.markdown("<ul>"+str1+"</ul>")
            st.markdown("""---""")

        is_noUpload=True
        if f_io is not None:
            fname_io = file_uploaded(f_io)
            is_noUpload=False
        else:
            fname_io=DEF_name_io
            file_notUploaded(DEF_name_io)

        if f_fec is not None:
            fname_fec = file_uploaded(f_fec)
            is_noUpload=False
        else:
            fname_fec=DEF_name_fec
            file_notUploaded(DEF_name_fec)

        if f_conv is not None:
            fname_conv = file_uploaded(f_conv)
            is_noUpload=False
        else:
            fname_conv=DEF_name_conv
            file_notUploaded(DEF_name_conv)

        if f_co2 is not None:
            fname_co2 = file_uploaded(f_co2)
            is_noUpload=False
        else:
            fname_co2=DEF_name_co2
            file_notUploaded(DEF_name_co2)

        #if f_agg is not None:
        #    fname_agg = file_uploaded(f_agg)
        #else:
        #    fname_agg=DEF_path_agg
        #    file_notUploaded(DEF_path_agg)

        #print(df1)
        if (not is_noUpload):
            #st.markdown("""---""")
            button = st.button('Recompute Matrix')
            #button = st.button('Recompute Matrix', on_click=calc_mat(fname_io, fname_fec, fname_conv, fname_co2, fname_agg))
            #button.on_click(calcmat(f_io, f_fec, f_conv, f_co2, f_agg))
        
            if (button): 
                
                calc_mat(fname_io, fname_fec, fname_conv, fname_co2)
                st.write("EEIO matrices have been recomputed")
            #print("")

with tab3:
    st.write("Indonesia Carbon Footprint Calculator")
    st.write("Version 0.0.1")
    st.write("September 2024")