import requests
import streamlit as st
from streamlit_lottie import st_lottie

import pandas as pd
import numpy as np
import pickle 

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title='Loan Prediction', page_icon=':moneybag:', layout='wide')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_money = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_xpmdb0zj.json')

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("./Project_Loan_Prediction/style/style.css")

# ---- HEADER SECTION ----
with st.container():
    
    st.subheader("Hi, I am Fadli :wave:")
    st.title("A Machine Learning Enthusiast")
    st.write(
        "I am passionate about finding ways to use Python and Machine Learning in business settings."
    )

with st.container():
    st.write('---')
    left_col, right_col = st.columns(2)
    with left_col:
        st.title('LOAN PREDICTION :moneybag:')
        st.subheader('Hello Everyone, this project is created for loan prediction')
        st.text('This project is using Random Forest Classifier algorithm.')
    with right_col:
        st_lottie(lottie_money, height=300, key="coding")



# load model
filename = 'model_smote_rf.sav'
loaded_model = pickle.load(open(filename, 'rb'))

st.write('---')
# section memasukkan / input feature
emp_status = st.selectbox(
    'Are you Employeed?',
    ('Yes', 'No')
)

st.write('You selected:', emp_status)

bank_balance = st.number_input('Input your current bank balance', 0)
salary = st.number_input('Input your current salary (per year)', 0)

btn_pred = st.button('Request Loan')

if btn_pred:
    emp_stat = 1 if emp_status == 'Yes' else 0
    
    tmp_pd = pd.DataFrame({
        'Employed':[emp_stat],
        'Bank Balance':[bank_balance],
        'Annual Salary':[salary]
    })

    res = loaded_model.predict(tmp_pd)
    if res == 1:
        st.write("Your Loan Request Accepted, We will contact you for further information")
    else:
        st.write('Your Loan Request Rejected')

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get In Touch With Me!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/rizaldyfadli77@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()

