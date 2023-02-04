import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np

app=Flask(__name__)
model=pickle.load(open('irisreg.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictapi',methods=['POST'])   
def predictapi():
    data=request.json['data']
    newdata=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(newdata)
    print(output[0])
    return jsonify(int(output[0]))



if __name__=="__main__":
    app.run(debug=True)