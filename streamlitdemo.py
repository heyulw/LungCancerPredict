import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn import preprocessing
import numpy as np


def get_clean_data():
  df = pd.read_csv("/content/drive/MyDrive/Data/dataset.csv")

  le=preprocessing.LabelEncoder()
  df['GENDER']=le.fit_transform(df['GENDER'])
  df['SMOKING']=le.fit_transform(df['SMOKING'])
  df['YELLOW_FINGERS']=le.fit_transform(df['YELLOW_FINGERS'])
  df['ANXIETY']=le.fit_transform(df['ANXIETY'])
  df['PEER_PRESSURE']=le.fit_transform(df['PEER_PRESSURE'])
  df['CHRONIC DISEASE']=le.fit_transform(df['CHRONIC DISEASE'])
  df['FATIGUE ']=le.fit_transform(df['FATIGUE '])
  df['ALLERGY ']=le.fit_transform(df['ALLERGY '])
  df['WHEEZING']=le.fit_transform(df['WHEEZING'])
  df['ALCOHOL CONSUMING']=le.fit_transform(df['ALCOHOL CONSUMING'])
  df['COUGHING']=le.fit_transform(df['COUGHING'])
  df['SHORTNESS OF BREATH']=le.fit_transform(df['SHORTNESS OF BREATH'])
  df['SWALLOWING DIFFICULTY']=le.fit_transform(df['SWALLOWING DIFFICULTY'])
  df['CHEST PAIN']=le.fit_transform(df['CHEST PAIN'])
  df['LUNG_CANCER']=le.fit_transform(df['LUNG_CANCER'])
  return df


def add_sidebar():
  st.sidebar.header('User Input Features')
  data = get_clean_data()
  slider_labels = [
        ("GENDER:0 tương ướng với nũ, 1 tương ứng với Nam", "gender"),
        ("AGE", "a"),
        ("SMOKING:0 tương ứng với không ,1 tương ứng với có ", "s"),
        ("YELLOW FINGERS:0 tương ứng với không ,1 tương ứng với có", "y"),
        ("ANXIETY:0 tương ứng với không ,1 tương ứng với có", "an"),
        ("PEER PRESSURE:0 tương ứng với không ,1 tương ứng với có", "p"),
        ("CHRONIC DISEASE:0 tương ứng với không ,1 tương ứng với có", "c"),
        ("FATIGUE:0 tương ứng với không ,1 tương ứng với có", "f"),
        ("ALLERGY:0 tương ứng với không ,1 tương ứng với có","all"),
        ("WHEEZING:0 tương ứng với không ,1 tương ứng với có", "w"),
        ("ALCOHOL CONSUMING:0 tương ứng với không ,1 tương ứng với có", "al"),
        ("COUGHING:0 tương ứng với không ,1 tương ứng với có", "co"),  
        ("SHORTNESS OF BREATH:0 tương ứng với không ,1 tương ứng với có", "sh"),
        ("SWALLOWING DIFFICULTY:0 tương ứng với không ,1 tương ứng với có", "sw"),   
        ("CHEST PAIN:0 tương ứng với không ,1 tương ứng với có", "ch"),
    ]
  input_dict = {}
  for label, key in slider_labels:
    if label == "AGE":
      input_dict[key] = st.sidebar.slider(
      label,
      min_value=18,
      max_value=80,    
    )
    else:
      input_dict[key] = st.sidebar.number_input(
        label,
        min_value=0,
        max_value=1,
      
      )
    
  return input_dict

def get_radar_chart(input_data): 
  categories = ['GENDER', 'AGE', 'SMOKING', 'YELLOW FINGERS', 
                'ANXIETY', 'PEER PRESSURE', 
                'CHRONIC DISEASE', 'FATIGUE',
                'ALLERGY', 'WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['a'],input_data['gender'], input_data['s'], input_data['y'],
          input_data['an'], input_data['p'], input_data['c'],
          input_data['f'], input_data['all'], input_data['w'],
          input_data['al'], input_data['co'], input_data['sh'], input_data['sw'], input_data['ch']
        ],
        theta=categories,
        fill='toself',
        
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig




  
  

def add_predictions(input_data):
  model = pickle.load(open("/content/drive/MyDrive/model.pkl", "rb"))
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  prediction = model.predict(input_array)
  st.subheader("Dự đoán ung thư phổi")
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Không có khả năng bị bệnh</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Có khả năng bị bệnh hãy đến gặp bác sĩ</span>", unsafe_allow_html=True)
    
def main():
  st.set_page_config(
    page_title="Dự đoán ung thư phổi",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("/content/drive/MyDrive/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    input_data = add_sidebar()
  
  with st.container():
    st.title("Dự đoán ung thư phổi")
    st.write("""
               Ung thư phổi là một loại ung thư  và thường xảy ra nhiều nhất ở những người hút thuốc. Nguyên nhân gây ra ung thư phổi bao gồm hút thuốc, hít phải khói thuốc từ người khác, tiếp xúc với một số chất độc hại và tiền sử gia đình. Các triệu chứng bao gồm ho (thường đi kèm với máu), đau ngực, sổ mũi và mất cân. Những triệu chứng này thường chỉ xuất hiện khi ung thư đã phát triển nặng. Các phương pháp điều trị thay đổi nhưng có thể bao gồm phẫu thuật, hóa trị, điều trị bằng tia X, điều trị bằng thuốc tập trung và điều trị miễn dịch.
              """)

  col1,col2= st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)
    
  
   


 
if __name__ == '__main__':
  main()