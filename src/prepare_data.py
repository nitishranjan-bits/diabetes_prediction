import pandas as pd
import os
from config import PROJECT_ROOT


def prepare_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'diabetes.csv')
    if not os.path.exists(data_path):
        # Create an empty DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                   'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    else:
        df = pd.read_csv(data_path)

    # You can add any data preparation steps here

    # Save the prepared data
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)


if __name__ == "__main__":
    prepare_data()