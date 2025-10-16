from flask import Flask, render_template, request, redirect, url_for, send_file, session
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)
app.secret_key = "your_secret"

# Function to generate dataset (10,000 points)
def generate_dataset(n=10000):
    load = np.random.randint(1000, 10000, n)
    fos = np.random.uniform(2.5, 5.0, n)
    ys = np.random.randint(300, 2000, n)
    # Calculate design dimensions as per textbook formulas
    d = np.sqrt( (4*load)/(np.pi*ys/fos) )    # Diameter of rod
    d1 = d                                   # Diameter of knuckle pin
    d2 = 2*d                                 # Outer diameter of eye
    d3 = 1.5*d                               # Diameter of pin head & collar
    t = 1.25*d                               # Thickness of single eye
    t1 = 0.75*d                              # Thickness of fork
    t2 = 0.5*d                               # Thickness of pin head
    # Induced stresses (for safety status, example formulas)
    rod_stress = load / ((np.pi/4)*d**2)
    pin_shear = load / (2 * (np.pi/4) * d1**2)
    eye_crush = load / (d1 * t)
    # Safety check (simplified)
    safe_rod = (rod_stress < ys/fos)
    safe_pin = (pin_shear < ys/fos)
    safe_eye = (eye_crush < ys/fos)
    df = pd.DataFrame({
        'Load': load,
        'YieldStrength': ys,
        'FactorOfSafety': fos,
        'RodDiameter': d,
        'PinDiameter': d1,
        'EyeDiameter': d2,
        'CollarDiameter': d3,
        'EyeThickness': t,
        'ForkThickness': t1,
        'PinHeadThickness': t2,
        'RodStress': rod_stress,
        'PinShearStress': pin_shear,
        'EyeCrushingStress': eye_crush,
        'RodSafe': safe_rod,
        'PinSafe': safe_pin,
        'EyeSafe': safe_eye
    })
    filename = "knuckle_joint_data.csv"
    df.to_csv(filename, index=False)
    return filename

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/generate', methods=['POST'])
def generate():
    filename = generate_dataset()
    return send_file(filename, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save('uploaded_knuckle_joint.csv')
    # Train model after upload
    df = pd.read_csv('uploaded_knuckle_joint.csv')
    X = df[['Load', 'YieldStrength', 'FactorOfSafety']]
    y = df[['RodDiameter', 'PinDiameter', 'EyeDiameter', 'CollarDiameter', 'EyeThickness', 'ForkThickness', 'PinHeadThickness',
            'RodStress', 'PinShearStress', 'EyeCrushingStress', 'RodSafe', 'PinSafe', 'EyeSafe']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'knuckle_joint_model.pkl')
    session['model_ready'] = True
    return redirect(url_for('predict'))

@app.route('/predict', methods=['GET'])
def predict():
    if not session.get('model_ready'):
        return redirect(url_for('home'))
    return render_template('predict.html')

@app.route('/do_predict', methods=['POST'])
def do_predict():
    load = float(request.form['load'])
    ys = float(request.form['yield_strength'])
    fos = float(request.form['fos'])
    model = joblib.load('knuckle_joint_model.pkl')
    input_data = np.array([[load, ys, fos]])
    result = model.predict(input_data)[0]
    # Prepare output labels
    out_labels = ["RodDiameter", "PinDiameter", "EyeDiameter", "CollarDiameter", "EyeThickness", "ForkThickness", "PinHeadThickness",
                  "RodStress", "PinShearStress", "EyeCrushingStress", "RodSafe", "PinSafe", "EyeSafe"]
    output = dict(zip(out_labels, result))
    return render_template('predict.html', result=output, input={'Load': load, 'YieldStrength': ys, 'FactorOfSafety': fos})

if __name__ == '__main__':
    app.run(debug=True)
