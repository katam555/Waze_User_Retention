from flask import Flask, render_template, request, jsonify
import csv
import pickle

app = Flask(__name__)

# Loading our pre-trained models
with open('logistic_regression_model.pkl', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

with open('naive_bayes_model.pkl', 'rb') as model_file:
    naive_bayes_model = pickle.load(model_file)

with open('random_forest_model.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

with open('xgboost_model.pkl', 'rb') as model_file:
    xgboost_model = pickle.load(model_file)

with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [
            float(request.form['sessions']),
            float(request.form['drives']),
            float(request.form['total_sessions']),
            float(request.form['n_days_after_onboarding']),
            float(request.form['total_navigations_fav1']),
            float(request.form['total_navigations_fav2']),
            float(request.form['driven_km_drives']),
            float(request.form['duration_minutes_drives']),
            float(request.form['activity_days']),
            float(request.form['driving_days'])
        ]

        # Use your pre-trained models for predictions
        logistic_regression_prediction = logistic_regression_model.predict([features])[0]
        naive_bayes_prediction = naive_bayes_model.predict([features])[0]
        random_forest_prediction = random_forest_model.predict([features])[0]
        xgboost_prediction = xgboost_model.predict([features])[0]
        svm_prediction = svm_model.predict([features])[0]
        knn_prediction = knn_model.predict([features])[0]

        # Create a dictionary of predictions
        predictions = {
            'logistic_regression': logistic_regression_prediction,
            'naive_bayes': naive_bayes_prediction,
            'random_forest': random_forest_prediction,
            'xgboost': xgboost_prediction,
            'svm': svm_prediction,
            'knn': knn_prediction,
        }

        return render_template('result.html', predictions=predictions)

    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            # Process the uploaded CSV file
            predictions = []

            try:
                # Opening the file in text mode ('rt') 
                with file.stream as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    header = next(csv_reader)  # As first row is the header

                    for row in csv_reader:
                        # Perform prediction logic for each row
                        features = [float(value) for value in row]
                        logistic_regression_prediction = logistic_regression_model.predict([features])[0]
                        naive_bayes_prediction = naive_bayes_model.predict([features])[0]
                        random_forest_prediction = random_forest_model.predict([features])[0]
                        xgboost_prediction = xgboost_model.predict([features])[0]
                        svm_prediction = svm_model.predict([features])[0]
                        knn_prediction = knn_model.predict([features])[0]

                        # Create a dictionary of predictions for each row
                        row_predictions = {
                            'logistic_regression': logistic_regression_prediction,
                            'naive_bayes': naive_bayes_prediction,
                            'random_forest': random_forest_prediction,
                            'xgboost': xgboost_prediction,
                            'svm': svm_prediction,
                            'knn': knn_prediction,
                        }

                        predictions.append(row_predictions)

            except csv.Error as e:
                return jsonify({'error': f'Error processing CSV file: {e}'})

            # Create a new CSV file with predictions
            output_file = 'predictions.csv'
            with open(output_file, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(header + ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Xgboost', 'SVM', 'KNN'])
                for row, row_predictions in zip(csv_reader, predictions):
                    csv_writer.writerow(row + [row_predictions['logistic_regression'],
                                               row_predictions['naive_bayes'],
                                               row_predictions['random_forest'],
                                               row_predictions['xgboost'],
                                               row_predictions['svm'],
                                               row_predictions['knn']])

            return jsonify({'success': True, 'output_file': output_file})

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
