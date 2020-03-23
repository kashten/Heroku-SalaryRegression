import numpy as np
from flask import Flask, request, jsonify, render_template #to post the model. render redirects to first homepage.
import pickle
from sklearn import preprocessing

app = Flask(__name__) #initialize the flask app.
model = pickle.load(open('model.pkl', 'rb')) #Load the pickled model. rb is read mode.
scaler = pickle.load(open('scaler.pkl','rb'))
#two functions are defined to 1. define where the root api should go and 2. 

@app.route('/')  #need to create this in flask for creating URI wrt to api.
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST']) #method is post. after providing three inputs, prediction is done.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] #this will get the input features from request.form (user input)
    final_features = [np.array(int_features)] #converts features into an array.
    final_features = preprocessing.Normalizer().fit_transform(final_features) #scale the features
    prediction = model.predict(final_features) #predicts the target from arrayed integer features 

    output = round(prediction[0], 2)  #get the output and round it off (answer) to two decimal places.

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output)) #prediction text is coming from the index.html 'prediction_text' tag.

#the third function is to pass on a hard coded json.
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__": #main function.
    app.run(debug=True)