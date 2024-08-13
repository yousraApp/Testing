from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import pandas as pd
framingham = pd.read_csv('framingham.csv')# Dropping null values
framingham = framingham.dropna()
framingham.head()

from sklearn.metrics import accuracy_score


# Assuming 'framingham' is your DataFrame and 'TenYearCHD' is a column in it
X = framingham.drop('TenYearCHD', axis=1)
#from sklearn.metrics import accuracy_scoreX = framingham.drop('TenYearCHD',axis=1)
# Assuming 'framingham' is your DataFrame and 'TenYearCHD' is a column in it
#X = framingham.drop('TenYearCHD', axis=1)
y = framingham['TenYearCHD']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling class imbalance with RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Define and train a model using a pipeline
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

import joblib
joblib.dump(pipeline, 'fhs_rf_model.pkl') 


import streamlit as st
import joblib
import pandas as pd

st.write("### 10 Year Heart Disease Prediction")

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])
#gender = st.selectbox("Enter your gender",["Male", "Female"])
col1, col2, col3 = st.columns(3)

gender = col1.selectbox("Enter your gender",["Male", "Female"])


age = col2.number_input("Enter your age")

education = col3.selectbox("Highest academic qualification",["High school", "Graduate degree", "Postgraduate degree", "PhD"])

isSmoker = col1.selectbox("Are you currently a smoker?",["Yes","No"])

yearsSmoking = col2.number_input("Number of daily cigarettes")

BPMeds = col3.selectbox("Are you currently on BP medication?",["Yes","No"])

stroke = col1.selectbox("Have you ever experienced a stroke?",["Yes","No"])

hyp = col2.selectbox("Do you have hypertension?",["Yes","No"])

diabetes = col3.selectbox("Do you have diabetes?",["Yes","No"])

chol = col1.number_input("Enter your cholesterol level")

sys_bp = col2.number_input("Enter your systolic blood pressure")

dia_bp = col3.number_input("Enter your diastolic blood pressure")

bmi = col1.number_input("Enter your BMI")

heart_rate = col2.number_input("Enter your resting heart rate")

glucose = col3.number_input("Enter your glucose level")

#st.button('Predict')

def transform(data):
    result = 3
    if data == 'High school':
        result = 0
    elif data == 'Graduate degree':
        result = 1
    elif data == 'Postgraduate degree':
        result = 2
    return result


df_pred = pd.DataFrame([[gender, age, education, isSmoker, yearsSmoking,BPMeds,stroke,hyp,diabetes,chol,sys_bp,dia_bp,bmi,heart_rate,glucose]],

columns= ['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'])

df_pred['male'] = df_pred['male'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['prevalentHyp'] = df_pred['prevalentHyp'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['prevalentStroke'] = df_pred['prevalentStroke'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['diabetes'] = df_pred['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['BPMeds'] = df_pred['BPMeds'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['currentSmoker'] = df_pred['currentSmoker'].apply(lambda x: 1 if x == 'Yes' else 0)
# Apply the transformation function to the 'education' column
df_pred['education'] = df_pred['education'].apply(transform)


model = joblib.load('fhs_rf_model.pkl')
prediction = model.predict(df_pred)




# Display result
if st.button('Predict'):
	if prediction[0] == 0:
		st.write('<p class="big-font">You likely will not develop heart disease in 10 years.</p>', unsafe_allow_html=True)
        #win32api.MessageBox(0, 'You likely will not develop heart disease in 10 years.', 'Heart Disease Prediction')
	else:
		st.write('<p class="big-font">You are likely to develop heart disease in 10 years.</p>', unsafe_allow_html=True)
        #win32api.MessageBox(0, 'You are likely to develop heart disease in 10 years.', 'Heart Disease Prediction')


##########################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data (replace this with your actual dataset)
# Load your dataset here. For demonstration, we'll create a mock dataframe.
# framingham = pd.read_csv('framingham.csv')  # Load your actual dataset

# Define the continuous features
continuous_features = ['age', 'BMI', 'totChol', 'heartRate', 'sysBP', 'diaBP']

# Identify the features to be converted to object data type
features_to_convert = [feature for feature in framingham.columns if feature not in continuous_features]

# Convert the identified features to object data type
framingham[features_to_convert] = framingham[features_to_convert].astype('object')

# Filter out continuous features for the univariate analysis
df_continuous = framingham[continuous_features]

# Set up the subplot
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Loop to plot histograms for each continuous feature
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3
    values, bin_edges = np.histogram(df_continuous[col],
                                     range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))

    graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                         edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Count', fontsize=12)
    ax[x, y].set_xticks(np.round(bin_edges, 1))
    ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
    ax[x, y].grid(color='lightgrey')

    for j, p in enumerate(graph.patches):
        ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                          ha='center', fontsize=10, fontweight="bold")

    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='#ff826e', edgecolor='white', pad=0.5))

# Hide the last subplot (if there are fewer than 6 features)
if len(df_continuous.columns) < 6:
    ax[1, 2].axis('off')

st.title('Distribution of Continuous Variables')
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Display the figure in Streamlit
st.pyplot(fig)

##########################################################   
#---------------------------------------------------------
# Set color palette
# Define the continuous features
continuous_features = ['age', 'BMI', 'totChol', 'heartRate', 'sysBP', 'diaBP']

# Set a custom color palette
sns.set_palette(['#ff826e', 'red'])

# Create the subplots
fig, ax = plt.subplots(len(continuous_features), 2, figsize=(15, 15), gridspec_kw={'width_ratios': [1, 2]})

# Loop through each continuous feature to create barplots and kde plots
for i, col in enumerate(continuous_features):
    # Barplot showing the mean value of the feature for each target category
    graph = sns.barplot(data=framingham, x="TenYearCHD", y=col, ax=ax[i, 0])

    # KDE plot showing the distribution of the feature for each target category
    sns.kdeplot(data=framingham[framingham["TenYearCHD"] == 0], x=col, fill=True, linewidth=2, ax=ax[i, 1], label='0')
    sns.kdeplot(data=framingham[framingham["TenYearCHD"] == 1], x=col, fill=True, linewidth=2, ax=ax[i, 1], label='1')
    ax[i, 1].set_yticks([])
    ax[i, 1].legend(title='Heart Disease', loc='upper right')

    # Add mean values to the barplot
    for cont in graph.containers:
        graph.bar_label(cont, fmt='         %.3g')

# Set the title for the entire figure
st.title('Continuous Features vs Target Distribution')
plt.tight_layout()

# Display the figure in Streamlit

 

# Display the figure in Streamlit
st.pyplot(fig)


#############################################################