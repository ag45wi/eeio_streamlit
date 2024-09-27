import streamlit as st

if 'IS_LOCAL' not in st.session_state:
    st.session_state['IS_LOCAL'] = 0  #1 for localhost; 0 for remote-host