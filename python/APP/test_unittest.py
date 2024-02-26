import unittest
import requests
import json
import os
import pandas as pd

#################################
#CE CODE TEST LA REPONSE DE L'API
#################################

class TestAPI(unittest.TestCase):
    def test_api_response(self):
        api_url = 'https://florianscoringapi-aec4d97f50b6.herokuapp.com/predictions'
        inputs = {'ID': 233721}
        response = requests.post(api_url, json=inputs)
        self.assertEqual(response.status_code, 200)

###########################################################################################################
#CE CODE S'ASSURE QUE LE DATAFRAME DE SIMULATION EST BIEN IMPLEMENTé PAR LES VALEURS FOURNIES PAR LE CLIENT
###########################################################################################################

current_dir = os.path.dirname(os.path.realpath(__file__))
sim_path = os.path.join(current_dir, "sim_df.csv")

def prepare_sim_df(filepath, age, enfant, ancieneté, genre, revenus, montant):
    # Charger le DataFrame
    sim_df = pd.read_csv(filepath)
    
    # Arrondir les valeurs du DataFrame
    sim_df = sim_df.round(0)
    
    # Supprimer les colonnes non nécessaires
    columns_to_drop = ['Unnamed: 0.1', 'TARGET']
    sim_df = sim_df.drop(columns=columns_to_drop)
    
    # Mise à jour des valeurs
    sim_df['DAYS_BIRTH'] = age
    sim_df["CNT_CHILDREN"] = enfant
    sim_df['DAYS_EMPLOYED'] = ancieneté
    sim_df['CODE_GENDER'] = 1 if genre == 'Femme' else 0
    sim_df['AMT_INCOME_TOTAL'] = revenus
    sim_df['AMT_CREDIT'] = montant
    
    return sim_df

class TestSimDfImplementation(unittest.TestCase):

    def test_prepare_sim_df(self):
        filepath = sim_path
        age = 30
        enfant = 2
        ancieneté = 5 
        genre = 'Femme'
        revenus = 50000
        montant = 250000

        sim_df = prepare_sim_df(filepath, age, enfant, ancieneté, genre, revenus, montant)

        self.assertIsNotNone(sim_df)
        self.assertIn('DAYS_BIRTH', sim_df.columns)

if __name__ == '__main__':
    unittest.main()