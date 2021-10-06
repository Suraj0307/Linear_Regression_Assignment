from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')
    # return "Yes we are live"


@app.route('/prediction', methods=['POST'])
def predict_page():
    model = pickle.load(open('LinearRegression.pickle', "rb"))
    scaler1 = pickle.load(open('scaler1.pickle', 'rb'))
    scaler2 = pickle.load(open('scaler2.pickle', 'rb'))
    if request.method == 'POST':
        user_input1 = float(request.form.get('input1'))
        user_input2 = float(request.form.get('input2'))
        user_input3 = float(request.form.get('input3'))
        user_input4 = float(request.form.get('input4'))
        user_input5 = float(request.form.get('input5'))
        user_input6 = float(request.form.get('input6'))
        user_input7 = float(request.form.get('input7'))
        user_input8 = float(request.form.get('input8'))
        user_input9 = float(request.form.get('input9'))
        user_input10 = float(request.form.get('input10'))
        user_input11 = float(request.form.get('input11'))
        user_input12 = float(request.form.get('input12'))
        prediction = model.predict(
            (scaler1.transform([[user_input1, user_input2, user_input3, user_input4, user_input5, user_input6,
                                 user_input7, user_input8, user_input9, user_input10, user_input11,
                                 user_input12]])))
        real_prdiction = scaler2.inverse_transform(prediction)
        return render_template("results.html", prediction=real_prdiction.flatten()[0])


if __name__ == '__main__':
    app.run(debug=True)
