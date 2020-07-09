import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open("model/chait.pkl", 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET'])
def main_get():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

@app.route("/", methods=['POST'])
def main_post():
    if flask.request.method == 'POST':
        # Extract the input
        age = flask.request.form['age']
        sex = flask.request.form['sex']
        cp = flask.request.form['cp']
        trestbps = flask.request.form['trestbps']
        chol = flask.request.form['chol']
        fbs = flask.request.form['fbs']
        restecg = flask.request.form['restecg']
        thalach = flask.request.form['thalach']
        exang = flask.request.form['exang']
        oldpeak = flask.request.form['oldpeak']
        slope = flask.request.form['slope']
        ca = flask.request.form['ca']
        thal = flask.request.form['thal']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                       columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
        print(prediction)
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('result.html',
                                     original_input={'age':age,
                                                     'sex':sex,
                                                     'cp':cp,
                                                     'trestbps':trestbps,
                                                     'chol':chol,
                                                     'fbs':fbs,
                                                     'restecg':restecg,
                                                     'thalach':thalach,
                                                     'exang':exang,
                                                     'oldpeak':oldpeak,
                                                     'slope':slope,
                                                     'ca':ca,
                                                     'thal':thal},
                                     result=int(prediction),
                                     )


if __name__ == '__main__':
    app.run()
