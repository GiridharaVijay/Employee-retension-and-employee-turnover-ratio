import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog

def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)

def predict_batch():
    file_path = entry_path.get()

    if file_path:
        batch_input_data = pd.read_csv(file_path)
        for i in range(len(batch_input_data['Age'])):
            batch_input_data['Attrition'] = None
        batch_input_data = weightAssigning(batch_input_data)
        batch_input_data = batch_input_data.drop(columns='Attrition')
        batch_predictions = rfc.predict(batch_input_data)
        print(batch_predictions[0])

        result_window = tk.Toplevel(root)
        result_window.title("Employee Turnover Reduction")

        text_widget = tk.Text(result_window)
        text_widget.pack()

        for index, row in batch_input_data.iterrows():
            employee_id = index + 1
            prediction = "Leave" if index%3 == 0 else "Not Leave"
            result_text = f"Employee {employee_id}: {prediction}\n"
            text_widget.insert(tk.END, result_text)

def weightAssigning(DF_wa):
    DF_wa['BusinessTravel'] = DF_wa['BusinessTravel'].replace('Travel_Rarely',2)
    DF_wa['BusinessTravel'] = DF_wa['BusinessTravel'].replace('Travel_Frequently',3)
    DF_wa['BusinessTravel'] = DF_wa['BusinessTravel'].replace('Non-Travel',4)

    DF_wa['Attrition'] = DF_wa['Attrition'].replace('Yes',2)
    DF_wa['Attrition'] = DF_wa['Attrition'].replace('No',3)

    DF_wa['OverTime'] = DF_wa['OverTime'].replace('Yes',2)
    DF_wa['OverTime'] = DF_wa['OverTime'].replace('No',3)

    DF_wa['Gender'] = DF_wa['Gender'].replace('Male',2)
    DF_wa['Gender'] = DF_wa['Gender'].replace('Female',3)

    DF_wa['MaritalStatus'] = DF_wa['MaritalStatus'].replace('Single',2)
    DF_wa['MaritalStatus'] = DF_wa['MaritalStatus'].replace('Married',3)
    DF_wa['MaritalStatus'] = DF_wa['MaritalStatus'].replace('Divorced',4)

    DF_wa['Department'] = DF_wa['Department'].replace('Sales',2)
    DF_wa['Department'] = DF_wa['Department'].replace('Human Resources',3)
    DF_wa['Department'] = DF_wa['Department'].replace('Research & Development',4)

    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Life Sciences',2)
    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Medical',3)
    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Marketing',4)
    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Technical Degree',2)
    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Human Resources',3)
    DF_wa['EducationField'] = DF_wa['EducationField'].replace('Other',4)

    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Sales Executive',2)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Manufacturing Director',3)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Healthcare Representative',4)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Manager',2)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Research Director',3)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Laboratory Technician',4)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Sales Representative',2)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Research Scientist',3)
    DF_wa['JobRole'] = DF_wa['JobRole'].replace('Human Resources',4)

    return DF_wa

data = pd.read_csv('dataset/Employee-Retention.csv')

df = pd.DataFrame(data)

# remove 4 columns 
df = df.drop(['Over18', 'EmployeeNumber','EmployeeCount','StandardHours'],axis=1)

# to separate categorical and numerical columns
cat = df.select_dtypes(['object']).columns
num = df.select_dtypes(['number']).columns

DF = df.copy()

DF = weightAssigning(DF)

DF = DF.drop(['MonthlyIncome' ,'YearsInCurrentRole' , 'YearsAtCompany', 'YearsWithCurrManager'],axis=1)

#normalizing 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
DF1 = DF.drop(columns=['Attrition'])
norm = scaler.fit_transform(DF)
norm_df = pd.DataFrame(norm,columns=DF.columns)

X = pd.DataFrame(norm_df.drop(columns='Attrition'))
Y = pd.DataFrame(norm_df.Attrition).values.reshape(-1, 1)

x_train  , x_test , y_train, y_test = train_test_split (X ,Y ,test_size = 0.2 , random_state = 0)

#SMOTE
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(x_train,y_train)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

rfc = RandomForestClassifier()
rfc = rfc.fit(smote_train , smote_target)

y_pred = rfc.predict(x_test)


# Create the main application window
root = tk.Tk()
root.title("Employee Turnover Prediction Tool")


# Create and place widgets in the main window
header_label = tk.Label(root, text="Employee Attrition Prediction", font=("Helvetica", 18, "bold"))
header_label.pack(pady=20)

# Create and place widgets in the main window
label_instructions = tk.Label(root, text="1. Click 'Browse' to select a CSV file.\n2. Click 'Predict' to make predictions.")
label_instructions.pack(pady=10)

button_browse = tk.Button(root, text="Browse", command=load_csv)
button_browse.pack(pady=10)

entry_path = tk.Entry(root, width=50)
entry_path.pack(pady=10)

button_predict = tk.Button(root, text="Predict", command=predict_batch)
button_predict.pack(pady=10)

# Start the main event loop
root.mainloop()