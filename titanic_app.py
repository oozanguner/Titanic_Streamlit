import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option ("display.max_columns", None)
pd.set_option ('display.expand_frame_repr', False)
pd.set_option ('display.width', 1000)

st.title (":ship: TITANIC SURVIVAL PREDICTION")
st.write ("""
 *
 The app predicts your survival probability if you had been on Titanic by using Random Forests algorithm and the real datas of Titanic's passengers.*""")

train = pd.read_pickle ("train.pkl")


def input_func(data):
    passengerId = np.random.randint (1, 891)
    ticket = "Ab32341"
    st.subheader ("Passenger Information")
    sex = st.selectbox ("Gender", ["Male", "Female"])
    status = st.selectbox ("Marital Status", ["Married", "Single"])
    pClass = st.selectbox ("Ticket Class", [1, 2, 3])
    fare = data.loc[data["Pclass"] == pClass]["Fare"].mean ()
    alone = st.selectbox ("Do you have any family members on Titanic?", ["Yes", "No"])
    cabin = st.selectbox ("Have you got any Cabin Number", ["Yes", "No"])
    embarked = st.selectbox ("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])
    age = st.slider ("Age", 0.5, 80.0, step=0.5)
    if alone == "Yes":
        sibSp = 1
        parch = 0
    else:
        sibSp = 0
        parch = 0

    return passengerId, ticket, sex, status, pClass, fare, age, sibSp, parch, cabin, embarked


passengerId, ticket, sex, status, pClass, fare, age, sibSp, parch, cabin, embarked = input_func (train)


def title_sex_formatting(age, sex, status):
    sex_format = sex.lower ()

    if age > 12 and sex == "Male":
        title_ = "Mr"
    elif age <= 12 and sex == "Male":
        title_ = "Master"
    elif status == "Married" and sex == "Female":
        title_ = "Mrs"
    elif status == "Single" and sex == "Female":
        title_ = "Miss"
    else:
        title_ = "Other"
    return sex_format, title_


sex_format, title_ = title_sex_formatting (age, sex, status)


def create_data_format(title_, sex_format):
    name_format = title_ + ". "
    df_new = pd.DataFrame ({"PassengerId": passengerId,
                            "Pclass": [pClass],
                            "Name": [name_format],
                            "Sex": [sex_format],
                            "Age": [age],
                            "SibSp": [sibSp],
                            "Parch": [parch],
                            "Ticket": [ticket],
                            "Fare": [fare],
                            "Cabin": [cabin],
                            "Embarked": [embarked]})

    df_new["Cabin"] = df_new["Cabin"].map ({"Yes": "C85",
                                            "No": np.nan})

    df_new["Embarked"] = df_new["Embarked"].map ({"Cherbourg": "C",
                                                  "Queenstown": "Q",
                                                  "Southampton": "S"})

    return df_new


df_new = create_data_format (title_, sex_format)

new_train = train.append (df_new)


def rare_encode_alt(dataframe, rare_ratio=0.01, threshold=10):
    rare_val_cols = [col for col in dataframe.columns if (dataframe[col].dtype == "O") & (
        (dataframe[col].value_counts () / len (dataframe) < rare_ratio).any ()) & (
                             dataframe[col].nunique () < threshold)]

    new_dataframe = dataframe.copy ()

    for i in rare_val_cols:
        tmp = new_dataframe[i].value_counts () / len (new_dataframe)
        rare_labels = tmp[tmp < rare_ratio].index
        new_dataframe[i] = np.where (new_dataframe[i].isin (rare_labels), 'Rare', new_dataframe[i])

    return new_dataframe


def titanic_prep(data):
    # Deleting Passenger Id variable
    dataframe = data.drop ("PassengerId", axis=1)

    # MISSING VALUES
    missing_cols = [col for col in dataframe.columns if (dataframe[col].isnull ().any ()) & (col != "Cabin")]
    for i in missing_cols:
        if i == "Age":
            dataframe[i].fillna (dataframe.groupby ("Pclass")[i].transform ("median"), inplace=True)
        elif dataframe[i].dtype == "O":
            dataframe[i].fillna (dataframe[i].mode ()[0], inplace=True)
        else:
            dataframe[i].fillna (dataframe[i].median (), inplace=True)

    # Examining "Cabin" if there is a cabin number or not and checking it that affect to "Survived" variable or not
    dataframe.loc[dataframe["Cabin"].notna (), "NEW_IsCabin"] = 1
    dataframe.loc[dataframe["Cabin"].isnull (), "NEW_IsCabin"] = 0

    # Deleting "Cabin" variable
    dataframe.drop ("Cabin", axis=1, inplace=True)

    # FEATURE ENGINEERING
    # Extracting title of passengers from "Name" Variable
    dataframe["NEW_TITLE"] = dataframe["Name"].str.extract ('([A-Za-z]+)\.', expand=False)

    # Aggregating rare values in "NEW_TITLE" variable
    new_df = rare_encode_alt (dataframe, 0.02, 20)

    # Creating "IsAlone" Feature
    new_df.loc[(new_df["Parch"] + new_df["SibSp"]) == 0, "NEW_isAlone"] = 1
    new_df.loc[(new_df["Parch"] + new_df["SibSp"]) > 0, "NEW_isAlone"] = 0
    new_df.drop (["Parch", "SibSp"], axis=1, inplace=True)

    # Passengers' Welfare Level
    # new_df["NEW_AGE_CAT"] = pd.cut (new_df["Age"], bins=[0, 20, 40, 60, 80],
    #                                 labels=[4, 3, 2, 1]).astype (int)
    new_df["NEW_PCLASS_SCORING"] = new_df["Pclass"].map ({1: 3, 2: 2, 3: 1})  # high value is better

    new_df["NEW_WELFARE_LEVEL"] = new_df["NEW_PCLASS_SCORING"] * new_df["Fare"] * 1 / new_df["Age"]

    # Encoding the columns that have more than two classes
    multiclass_cat_cols = [col for col in new_df.columns if 25 > new_df[col].nunique () > 2]

    new_df = pd.get_dummies (data=new_df, columns=multiclass_cat_cols, drop_first=False)

    # Encoding the columns that have two classes
    new_df["NEW_isMALE"] = new_df["Sex"].map ({"male": 1, "female": 0})
    new_df = new_df[[col for col in new_df.columns if new_df[col].dtype != "O"]]

    return new_df


df_prep = titanic_prep (new_train).reset_index (drop=True)

submission = df_prep.tail (1)
submission.drop (["Survived"], axis=1, inplace=True)

rfc_from_joblib = joblib.load ('tuned_rfc.pkl')

life_prob = rfc_from_joblib.predict_proba (submission)[:, 1][0] * 100


def click_button(life_prob):
    if 0 <= life_prob < 30:
        emoji = ":scream:"
    elif 30 <= life_prob < 50:
        emoji = ":frowning:"
    elif life_prob == 50:
        emoji = ":worried:"
    elif 60 >= life_prob > 50:
        emoji = ":relieved:"
    elif 80 >= life_prob > 60:
        emoji = ":sweat_smile:"
    else:
        emoji = ":sunglasses:"

    if st.button (""" Click Here To See Your Life Expectancy !!!"""):
        st.markdown ("""# *Your Life Expectancy :* % {:.2f} {}""".format (life_prob, emoji))


click_button (life_prob)
