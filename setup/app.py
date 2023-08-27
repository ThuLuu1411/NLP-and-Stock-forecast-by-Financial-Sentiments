from flask import Flask
from flask_restful import Resource, Api
from flask import Flask, render_template, request
from predict_text import predict_text

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
	if request.method == 'POST':
		im_dict = predict_text(request.form['username'])
		return(f'''<h1>{im_dict[0]}</h1>
                 <h1>{im_dict[1]}</h1>''')
	return render_template('home.html')