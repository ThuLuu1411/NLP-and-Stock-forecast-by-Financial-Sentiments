from flask import Flask
from flask_restful import Resource, Api
import NLP
import pickle
import pandas as pd
import os
import numpy as np

# model = pickle.load(open('finalized_model.sav', 'rb'))

# importing required modules
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "finalized_model.sav.zip"
  
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    model = pickle.load(zip.open('finalized_model.sav', 'r'))

def predict_text(text):
    clean_text = NLP.clean_text(text)
    remove = ['kinh_tế', 'thị_trường']
    predict_prob = model.predict_proba(pd.DataFrame({i: [(i in clean_text) and (i not in remove)] for i in model.feature_names_in_}))
    predict_prob = int(predict_prob[0][1]*100)
    if predict_prob >= 50:
        # print('Khuyến nghị giá sẽ giảm')
        noti1 = 'Khuyến nghị giá sẽ giảm' 
        noti2 = 'Sắc thái tiêu cực là: '+ str(predict_prob)+ '%'
        # print(noti1)
        # print(noti2)
        return (noti1, noti2)
    else:
        noti1 = 'Khuyến nghị giá vẫn ổn định' 
        noti2 = 'Sắc thái tiêu cực là: '+ str(predict_prob)+ '%'
        # print(noti1)
        # print(noti2)
        return (noti1, noti2)

# text = input('Nhập vào tiêu đề bài báo: ')
# while text != 'close':
#     predict_text(text)
#     text = input('Nhập vào tiêu đề bài báo: ')

# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#  	if request.method == 'POST':
#          im_dict = predict_text(request.form['username'])
#          # print(im_dict)
#          return(f'''<h1>{im_dict[0]}</h1>
#                     <h1>{im_dict[1]}</h1>''')
#  	return render_template('home.html')

app = Flask(__name__)
@app.route('/user/<text>')
def home(text):
    im_dict = predict_text(text)
    return f'{im_dict[0]} - {im_dict[1]}'



# if __name__ == '__main__':
#  	app.run()#host="0.0.0.0", port=80, debug=True)


