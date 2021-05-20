import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bs4 import BeautifulSoup
import re

def preprocesado(texto):
  # Se eliminan las etiquetas HTML mediante la librería BeautifulSoup
  soup = BeautifulSoup(texto, "html.parser")
  # Se convierte a minúsculas
  texto = soup.get_text().lower()
  #Se eliminan los enlaces
  texto = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', texto)
  # Unificar comillas
  texto = re.sub(r'([\'\“\”])', r'"', texto)
  # Se eliminan caracteres especiales (se sustituyen por un espacio)
  texto = re.sub(r'([\\\/\;\:\|•«\n])', ' ', texto)  
  # Se eliminan los espacios consecutivos (se han generado algunos previamente)
  texto = re.sub(r'\s+', ' ', texto).strip()
  return texto

def write_header():
    st.title('Generador de titulares')
    st.markdown('''
        - El titular se genera a partir del cuerpo.  
        - Se debe esperar a que los diversos componentes se carguen ()
    ''')

def write_ui():
    input = st.text_input('Introduce el cuerpo de la noticia y pulsa Enter', value="India's national language Hindi, and Pakistan's national language Urdu are almost the same.")
    if not input:
        return
    st.write(input)





st.set_page_config(page_title='Generador', layout='wide')
write_header()
#write_ui()

model_checkpoint= "josmunpen/mt5-small-spanish-summarization"
st.write(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
st.write("Tokenizer cargado.")


loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
st.write("Modelo cargado.")
    
entrada = st.text_input('Introduce el cuerpo de la noticia y pulsa Enter')
if st.button('Generar titular'):
    entrada = preprocesado(entrada)

    inputs = tokenizer.encode("summarize: " + entrada, return_tensors="pt", max_length=1024, truncation=True)
    outputs = loaded_model.generate(
        inputs, 
        max_length=128, 
        min_length=10, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    st.write(tokenizer.decode(outputs[0]))
