from flask import Flask,request,jsonify
import pickle
import numpy as np
import pathlib
from flask_cors import CORS

pt = pickle.load(open('artifact\pt.pkl','rb'))
vehicles = pickle.load(open(r'artifact\vehicles.pkl','rb'))
similarity_scores = pickle.load(open('artifact\similarity_scores.pkl','rb'))

#print(pt)

def recommend(vehicle_name):
    # index fetch
    index = np.where(pt.index==vehicle_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = vehicles[vehicles['vehicle.model'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('vehicle.model')['vehicle.model'].values))
        item.extend(list(temp_df.drop_duplicates('vehicle.model')['fuelType'].values))
        item.extend(list(temp_df.drop_duplicates('vehicle.model')['vehicle.make'].values))
        
        data.append(item)
    
    return data

#dataa =recommend("Xterra")
#print(dataa)

app = Flask(__name__)
CORS(app)


@app.route('/predict',methods=['POST'])
def recommendation():
    vehicle = request.json['vehicle']
    data = recommend(vehicle)
    
    return jsonify(recommend=data)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4000,debug=True)
