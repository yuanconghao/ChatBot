import streamlit as st


#Config
st.set_page_config(layout="wide", page_icon="💬", page_title="ChatBot")


#Title
st.markdown(
    """
    <h2 style='text-align: center;'>ChatBot, your data-aware assistant 🤖</h1>
    """,
    unsafe_allow_html=True,)

st.markdown("---")


#Description
st.markdown(
    """ 
    <h5 style='text-align:center;'>I'm an intelligent ChatBot. I use large language models to provide
    context-sensitive natural language interactions. My goal is to help you better understand your data.
    I support PDF, TXT, and CSV data, with more coming soon! 🧠</h5>
    """,
    unsafe_allow_html=True)





