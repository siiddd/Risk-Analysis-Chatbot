import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
import pandas as pd
import nltk

pickledmodel = open(r'C:\Users\siddhn\Desktop\model_RF.pkl',"rb")
nlp_classifier = pickle.load(pickledmodel)

f = open(r'C:\Users\siddhn\Desktop\glove.6B.200d.txt', 'r', encoding = 'utf-8')

def brazil(problem):
  problem = problem.lower()
  embeddings = {}

  for line in f:
    values = line.split()
    words = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings[words] = coefs
  f.close()

  data_list = list()
  sentence = np.zeros(200)
  count = 0
  dataset_word = nltk.word_tokenize(problem)
  for x in dataset_word:
       try:
           sentence += embeddings[x]
           count += 1
       except KeyError:
           continue
  data_list.append(sentence / count)
  return data_list

def final(x):
    x = pd.DataFrame(x).T
    prediction = nlp_classifier.predict(pd.DataFrame(x).T)
    return prediction

def main():
    st.title("Risk Analysis Chatbot")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    problem = st.text_input("Please type your problem elaborately.","Type your problem here")

    result=""

    if st.button("Predict the Accident Level"):
        x = brazil(problem)
        result = final(x)
        result = str(result)
        result = result[1]
    st.success('The Accident Level is  {}'.format(result))

if __name__=='__main__':
    main()