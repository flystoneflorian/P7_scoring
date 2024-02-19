import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import re
import requests
import math
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import json
import joblib
import seaborn as sns
from streamlit_shap import st_shap
import shap
from shap import TreeExplainer
from catboost import Pool
from shap.maskers import Independent
import os
shap.initjs()

st.set_page_config(page_title="Probabilité de remboursement de crédit", layout="wide")
st.markdown("<h1 style='text-align: center; color: #5A5E6B;'>Probabilité de remboursement de crédit</h1>", unsafe_allow_html=True)

application_train = pd.read_csv("C:/P7_git/python/APP/application_train.csv")
application_test = pd.read_csv("C:/P7_git/python/APP/application_test.csv")
# Chargement des données
df = pd.read_csv("C:/P7_git/python/APP/test_api.csv")
columns_to_drop = ['TARGET', 'Unnamed: 0.1']
df=df.drop(columns=columns_to_drop)
df["SK_ID_CURR"]=df["SK_ID_CURR"].convert_dtypes()
sk=df["SK_ID_CURR"]
df.index=sk
df_shap=df.copy()


# Chargement du modèle
model = joblib.load(r'/P7_git/python/second_best_model.joblib')

# Prétraitement et feature engineering
def feature_engineering(df):
    new_df = pd.DataFrame()
    new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: -x / 365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x: int(-x / 365.25))
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df

def get_shap_values(model, df_transformed, client_id):
    explainer = shap.Explainer(model, df_transformed)
    client_data = df_transformed.loc[[client_id]]
    shap_values = explainer.shap_values(client_data)
    return explainer, shap_values, client_data

def get_shap_global(model, df_transformed):
    explainer = shap.Explainer(model, df_transformed)
    shap_values = explainer.shap_values(df_transformed)
    return explainer, shap_values

def graphique(df,feature, features_client, title):

      if (not (math.isnan(features_client))):
            fig = plt.figure(figsize = (10, 4))
            t0 = df.loc[df['TARGET'] == 0]
            t1 = df.loc[df['TARGET'] == 1]
            sns.kdeplot(t0[feature].dropna(), label = 'Bon client', color='g')
            sns.kdeplot(t1[feature].dropna(), label = 'Client à risque', color='r')
            plt.axvline(float(features_client), color="blue", 
                        linestyle='--', label = 'Position Client')

            plt.title(title, fontsize='20', fontweight='bold')
            plt.legend()
            plt.show()  
            st.pyplot(fig)
      else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")

def email_valide(email):
    regex = regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    return re.match(regex, email, re.IGNORECASE) is not None

def calculer_age(date_naissance):
    aujourd_hui = date.today()
    age = aujourd_hui.year - date_naissance.year - ((aujourd_hui.month, aujourd_hui.day) < (date_naissance.month, date_naissance.day))
    return age

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def on_run_clicked():
    st.session_state['run_clicked'] = True

features_à_selectionner = {
    'CODE_GENDER': "GENRE",
    'DAYS_BIRTH': "AGE",
    'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
    'CNT_CHILDREN': "NB ENFANTS",
    'FLAG_OWN_CAR': "POSSESSION VEHICULE",
    'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
    'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
    'OCCUPATION_TYPE': "EMPLOI",
    'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
    'AMT_INCOME_TOTAL': "REVENUS",
    'AMT_CREDIT': "MONTANT CREDIT",
    'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
    'AMT_ANNUITY': "MONTANT_ANNUITES",
    'NAME_INCOME_TYPE': "TYPE REVENUS",
    'EXT_SOURCE_1': "EXT_SOURCE_1",
    'EXT_SOURCE_2': "EXT_SOURCE_2",
    'EXT_SOURCE_3': "EXT_SOURCE_3",
}

features_shap_à_selectionner = {
    'CODE_GENDER': "GENRE",
    'DAYS_BIRTH': "AGE",
    'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
    'CNT_CHILDREN': "NB ENFANTS",
    'FLAG_OWN_CAR': "POSSESSION VEHICULE",
    'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
    'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
    'OCCUPATION_TYPE': "EMPLOI",
    'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
    'AMT_INCOME_TOTAL': "REVENUS",
    'AMT_CREDIT': "MONTANT CREDIT",
    'PAYMENT_RATE': "DUREE CREDIT",
    'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
    'AMT_ANNUITY': "MONTANT_ANNUITES",
    'NAME_INCOME_TYPE': "TYPE REVENUS",
    'EXT_SOURCE_1': "EXT_SOURCE_1",
    'EXT_SOURCE_2': "EXT_SOURCE_2",
    'EXT_SOURCE_3': "EXT_SOURCE_3",
}

def feature_engineering(df):
    new_df = pd.DataFrame()
    new_df = df.copy()
    new_df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 'Femme' if x == 1 else 'Homme')
    new_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x : -x/365.25)
    new_df['DAYS_BIRTH'] = df['DAYS_BIRTH'].apply(lambda x : int(-x/365.25))
    new_df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 'Oui' if x == 1 else 'Non')
    return new_df


#les informations relatives aux clients
features_numériques_à_selectionner={
      'CNT_CHILDREN': "NB ENFANTS",
      'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
      'AMT_INCOME_TOTAL': "REVENUS",
      'AMT_CREDIT': "MONTANT CREDIT", 
      'AMT_ANNUITY': "MONTANT ANNUITES",
      'EXT_SOURCE_1': "EXT_SOURCE_1",
      'EXT_SOURCE_2': "EXT_SOURCE_2",
      'EXT_SOURCE_3': "EXT_SOURCE_3",
}

default_list=\
["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","REVENUS","MONTANT CREDIT"]
numerical_features = [ 'DAYS_BIRTH','CNT_CHILDREN', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

col1, col2 = st.columns(2)

with col1:
    if 'run_clicked' not in st.session_state:
        st.session_state['run_clicked'] = False
    st.sidebar.header("**Identifiant**")
    st.header("Client")
    id_filter = st.sidebar.selectbox("Entrez identifiant client", pd.unique(sk))

    inputs= {'ID':int(id_filter)}
    data_json=json.dumps(inputs)

    risque = st.sidebar.checkbox('Risque de defaillance sur le crédit', key='risque_defaillance')

    api_url = 'http://127.0.0.1:8000/predictions'
    data = {"ID: int(id_filter)"}
    #res = requests.post(url=api_url, data=json.dumps(inputs))
    res = requests.post(url=api_url, json=inputs)
    #st.subheader(f"Reponse from API ={res.text}")


    infos_client = st.sidebar.checkbox("Afficher les informations client ", key='show_client_info')
    show_all_info = st.sidebar.checkbox("Afficher toutes les informations (dataframe brute)", key='show_all_info')
    infos_graphique = st.sidebar.checkbox("Afficher les données qui ont influencé le calcul de son score ", key='show_client_graph')
    comparaison = st.sidebar.checkbox('Comparaison aux autres clients')
    number = st.sidebar.slider('Sélectionner le nombre de features à afficher', min_value=2, max_value=10, value=6)
    run_button =  st.sidebar.button('Run', on_click=on_run_clicked)

    if st.session_state['run_clicked']:   
        # Actions à exécuter lorsque le bouton Run est cliqué
        if risque:
            if res.status_code == 200:
                score=res.json()['score']
                pred=res.json()['prediction']

            if pred == 0:
                st.success(f"Probabilité : {score}")
                st.markdown("<h2 style='text-align: center; color: #44be6e;'>Crédit sans risque</h2>", unsafe_allow_html=True)
            else:
                st.error(f"Probabilité: {score}")
                st.markdown("<h2 style='text-align: center; color: #ff3d41;'>Crédit à risque</h2>", unsafe_allow_html=True)

        if infos_client:
            st.markdown("<h1 style='text-align: center; color: #5A5E6B; font-size: 24px;'>Informations relatives au client</h1>", unsafe_allow_html=True)
            df = feature_engineering(application_test)
            X = df[df["SK_ID_CURR"] == int(id_filter)]
            with st.spinner('Chargement des informations relatives au client...'):
                personal_df = X[list(features_à_selectionner.keys())]
                personal_df.rename(columns=features_à_selectionner, inplace=True)
                filtered = st.multiselect("Choisir les informations à afficher", 
                                            options=list(personal_df.columns), 
                                            default=list(default_list))
                df_info = personal_df[filtered]
                df_info['SK_ID_CURR'] = X.index.to_list()
                df_info = df_info.set_index('SK_ID_CURR')
                st.table(df_info.astype(str).T)
        
        if show_all_info:
            st.markdown("<h1 style='text-align: center; color: #5A5E6B; font-size: 24px;'>Afficher toutes les infos relatives au clients (dataframe brut)</h1>", unsafe_allow_html=True)
            df = feature_engineering(df)
            X = df[df["SK_ID_CURR"] == int(id_filter)]
            st.dataframe(X)

        if infos_graphique:
            st.markdown("<h1 style='text-align: center; color: #5A5E6B; font-size: 24px;'>Variables qui ont le plus influencé le score du client (en bleu positivement, en rouge négativement)</h1>", unsafe_allow_html=True)
            #id_ = df[df["SK_ID_CURR"] == int(id_filter)]
            fig, ax = plt.subplots(figsize=(15, 15))
            scaler = model.named_steps['scaler']
            df_transformed = pd.DataFrame(scaler.transform(df_shap), columns=df_shap.columns, index=df_shap.index)
            lgbmc_model = model.named_steps['LGBMC']
            client_id = int(id_filter)
            explainer, shap_values, client_data = get_shap_values(lgbmc_model, df_transformed, client_id)
            df_shap.rename(columns=features_shap_à_selectionner, inplace=True)  
            st_shap(shap.force_plot(explainer.expected_value, shap_values, client_data, feature_names=df_shap.columns.tolist())) 
            st_shap(shap.summary_plot(shap_values, client_data, feature_names=df_shap.columns.tolist(), max_display = number))  
        if comparaison:
            st.markdown("<h1 style='text-align: center; color: #5A5E6B; font-size: 24px;'>Comparaison aux autres clients</h1>", unsafe_allow_html=True)
            explainer, shap_values = get_shap_global(lgbmc_model, df_transformed)
            st_shap(shap.summary_plot(shap_values, df_transformed, feature_names=df_shap.columns.tolist(), max_display = number))
            ap_train = feature_engineering(application_train)
            # Actions à exécuter après que "Run" a été activé
            var = st.selectbox("Sélectionner une variable", list(features_numériques_à_selectionner.values()), index=0, key='var_selector')
            feature = list(features_numériques_à_selectionner.keys())[list(features_numériques_à_selectionner.values()).index(var)]
            X = feature_engineering(X)
            features_client_value = X[feature].iloc[0] if not X[feature].empty else np.nan
            title = f"Distribution de {var} parmi tous les clients avec position du client actuel"
            graphique(ap_train, feature, features_client_value, title)       


with col2:
    st.header('Simulation pour prospect')

    enfants = application_test["CNT_CHILDREN"].unique()
    application_test['CODE_GENDER'] = application_test['CODE_GENDER'].apply(lambda x: 'Femme' if x == 'F' else 'Homme')
    sex = application_test["CODE_GENDER"].unique()
    #application_test['DAYS_EMPLOYED'] = application_test['DAYS_EMPLOYED'].apply(lambda x: -x / 365.25)
    ancieneté = list(range(0, 51))
    #application_test['AMT_INCOME_TOTAL']

    nom = st.text_input("Nom et Prénom")
    sex = st.selectbox("Genre", sex)
    email = st.text_input("e-mail")
    if email:
        if email_valide(email):
            st.success("format de l'e-mail valide.")
        else:
            st.error("L'email n'est pas dans un format valide.")
    # Calcul de l'âge
    date_naissance = st.date_input("Entrez votre date de naissance", min_value=date(1924, 1, 1), max_value=date.today())
    age = calculer_age(date_naissance)
   # Vérifier si l'âge est au moins 18
    if age < 18:
        st.error("Vous devez avoir au moins 18 ans.")
    else:
        st.write(f"Vous avez : {age} ans")

    enfant = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1)
    revenus = st.number_input("Revenus en €")
    ancieneté = st.selectbox("Ancieneté en années", ancieneté)
    montant = st.number_input("Montant du prêt en €")

    if st.button("Calculer la probabilité d'accorder un crédit"):

        sim_df=pd.read_csv(r'C:\cygwin64\openclassrooms\Projet_7\env\python\Notebook\sim_df.csv')
        sim_df=sim_df.round(0)
        columns_to_drop = [ 'Unnamed: 0.1', 'TARGET']
        sim_df=sim_df.drop(columns=columns_to_drop)


        sim_df['DAYS_BIRTH'] = age
        sim_df["CNT_CHILDREN"] = enfant
        sim_df['DAYS_EMPLOYED'] = ancieneté        
        sim_df['CODE_GENDER'] = application_test['CODE_GENDER'].apply(lambda x: 1 if x == 'Femme' else 0)
        sim_df['AMT_INCOME_TOTAL'] = revenus
        sim_df['AMT_CREDIT'] = montant

        pred = model.predict(sim_df)[0]
        score = round(model.predict_proba(sim_df)[0][1], 2)

        if pred == 0:
            st.success(f"Probabilité : {score}")
            st.markdown("<h2 style='text-align: center; color: #44be6e;'>Crédit sans risque</h2>", unsafe_allow_html=True)
        else:
            st.error(f"Probabilité: {score}")
            st.markdown("<h2 style='text-align: center; color: #ff3d41;'>Crédit à risque</h2>", unsafe_allow_html=True)

