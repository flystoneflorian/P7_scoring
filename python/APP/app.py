
# 1. Library imports
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

#data
df=pd.read_csv('test_api.csv')
df["SK_ID_CURR"]=df["SK_ID_CURR"].convert_dtypes()
sk=df["SK_ID_CURR"]
df.index=sk
X=df.copy()
columns_to_drop = ['TARGET', 'Unnamed: 0.1']
X.drop(columns=columns_to_drop,inplace=True) 
#X.drop(columns=["SK_ID_CURR"],inplace=True) 

#model
#path = 'C:\\P7_git\\python\\APP\\'
model = joblib.load('second_best_model.joblib')

# 2. Create the app object
app = FastAPI()
class Item(BaseModel):
    ID: int

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
async def root():
    return {'message': 'Modèle de scoring'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.post('/predictions')
async def predictions(input:Item):

    prediction= model.predict(X[X.index == input.ID]).tolist()[0]
    score= model.predict_proba(X[X.index == input.ID])[:,1]
    #return int(round(score[0],3))
    return {'prediction': int(prediction), 'score':(round(score[0],3))
            }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
