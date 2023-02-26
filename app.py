import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
plt = platform.system()
if plt == 'Linnux': pathlib.Windows = pathlib.PosixPath
temp =  pathlib.PosixPath


#title
st.title('Transport(Boat, Airplane, Car) Classificator')

#upload a img
file= st.file_uploader('Upload a picture', type=['jpeg'])
if file:
    st.image(file)
    #image convertion
    img = Image.open(file)
    # do stuff with `img`

    output = io.BytesIO()
    img.save(output, format='JPEG')  # or another format
    output.seek(0)

    #model
    model = load_learner('transport_model.pkl')

    #prediction

    pred, pred_id, probs =  model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Accuracy: {probs[pred_id]*100:.1f}%')

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
