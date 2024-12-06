from sklearn.linear_model import LinearRegression
import flask 
from flask import render_template, jsonify
import pandas as pd
import random
import numpy as np

def generate_random_color():
    return f'rgba({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)}, 1)'

app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html.jinja')

@app.route('/data')
def data():
    data = pd.read_csv('data.csv',delimiter=';')
    data_list = data.to_dict(orient='records')
    return render_template('data.html.jinja', data=data_list)

@app.route('/predict')
def predict():
    return render_template('predict.html.jinja')

@app.route('/data/chart')
def chart():
    # Read the CSV file
    data = pd.read_csv('data.csv', delimiter=';')

    # Convert 'tanggal' to datetime, let pandas infer the format and handle mixed formats
    data['tanggal'] = pd.to_datetime(data['Tanggal'], format='%m/%d/%Y', errors='coerce')

    # Create a new column 'month_year' by extracting the month and year
    data['month_year'] = data['tanggal'].dt.to_period('M')

    # Group by 'month_year' and 'Keterangan', then count occurrences
    count_data = data.groupby(['month_year', 'Keterangan']).size().reset_index(name='count')

    # Convert the 'month_year' Period to string for serialization
    count_data['month_year'] = count_data['month_year'].astype(str)

    # Get the list of unique month_years for labels
    month_years = sorted(count_data['month_year'].unique())

    # Prepare the final result
    result = {
        "labels": month_years,  # The labels will be the unique month_years
        "datasets": []
    }

    # For each 'Keterangan', we prepare a dataset
    for keterangan in count_data['Keterangan'].unique():
        dataset = {
            "label": [keterangan],  # 'label' will contain the 'Keterangan'
            "data": [],
            "borderColor": generate_random_color(),  # Unique line color
            "backgroundColor": generate_random_color(),  # Unique background color
            "fill": False  # No fill for line chart
        }

        # Get the counts for each month_year
        for month_year in month_years:
            count = count_data[(count_data['month_year'] == month_year) & (count_data['Keterangan'] == keterangan)]['count'].sum()
            dataset['data'].append(int(count))  # Convert to int for JSON serializability
        
        result['datasets'].append(dataset)

    # Return the result as a Flask JSON response
    return jsonify(result)


@app.route('/predict/chart/<int:months>')
def predict_chart(months):
    # Read the CSV file
    data = pd.read_csv('data.csv', delimiter=';')

    # Convert 'tanggal' to datetime
    data['tanggal'] = pd.to_datetime(data['Tanggal'], format='%m/%d/%Y', errors='coerce')

    # Create a new column 'month_year' by extracting the month and year
    data['month_year'] = data['tanggal'].dt.to_period('M')

    # Group by 'month_year' and 'Keterangan', then count occurrences
    count_data = data.groupby(['month_year', 'Keterangan']).size().reset_index(name='count')

    # Convert the 'month_year' Period to string for serialization
    count_data['month_year'] = count_data['month_year'].astype(str)

    # Get the list of unique month_years for labels
    month_years = sorted(count_data['month_year'].unique())

    # Add labels for the predicted months
    last_month = pd.Period(month_years[-1], freq='M')
    for _ in range(months):
        last_month += 1
        month_years.append(last_month.strftime('%Y-%m'))

    result = {
        "labels": month_years,  # Include the predicted months' labels
        "datasets": []
    }

    # For each 'Keterangan', prepare a dataset
    for keterangan in count_data['Keterangan'].unique():
        dataset = {
            "label": keterangan,  # Label for the category
            "data": [],
            "borderColor": generate_random_color(),  # Unique line color
            "backgroundColor": generate_random_color(),  # Unique background color
            "fill": False  # No fill for line chart
        }

        # Historical data for linear regression
        historical_data = []
        for month_year in month_years[:-months]:
            count = count_data[(count_data['month_year'] == month_year) & (count_data['Keterangan'] == keterangan)]['count'].sum()
            dataset['data'].append(int(count))  # Append historical counts
            historical_data.append(count)

        # Predict next values iteratively
        if len(historical_data) > 1:  # Ensure sufficient data for prediction
            # Prepare data for linear regression
            X = np.arange(len(historical_data)).reshape(-1, 1)
            y = np.array(historical_data)
            model = LinearRegression()
            model.fit(X, y)

            # Iteratively predict for the specified months
            next_values = []
            for i in range(months):
                next_index = len(historical_data) + i
                predicted_value = model.predict([[next_index]])[0]
                predicted_value = max(predicted_value, 0)  # Ensure non-negative
                next_values.append(predicted_value)
                historical_data.append(predicted_value)  # Use for subsequent predictions

            dataset['data'].extend(map(int, next_values))
        else:
            # Use the average as the prediction if not enough data
            next_values = [np.mean(historical_data)] * months if historical_data else [0] * months
            dataset['data'].extend(map(int, next_values))

        result['datasets'].append(dataset)

    # Return the result as a Flask JSON response
    return jsonify(result)