import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from helpers.data_cleaning import getDataset, filterCategory, filterLocation
from helpers.sims import SVDTopMuseums

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    # episodes_df = pd.DataFrame(data['episodes'])
    all_museums_df = getDataset()
    # reviews_df = pd.DataFrame(data['reviews'])

app = Flask(__name__)
CORS(app)

@app.route("/") #route to url
def home():
    return render_template('base.html',title="sample html")

@app.route("/museums")
def get_museums():
    text = request.args.get("query")
    categories = request.args.getlist("categories")
    locations = request.args.getlist("locations")
    filtered_museums = set(filterCategory(categories))
    filtered_museums &= set(filterLocation(locations))
    print(categories)
    return json.dumps(SVDTopMuseums(text, filtered_museums))

@app.route("/locations")
def locations():
    state_city_map = {}
    for _, row in all_museums_df.iterrows():
        state = row['State']
        city = row['City']
        state_city_map.setdefault(state, set()).add(city)

    # sort by alphabetical order
    for state in state_city_map:
        state_city_map[state] = sorted(state_city_map[state])
    sorted_state_city_map = {state: state_city_map[state] for state in sorted(state_city_map)}

    return json.dumps(sorted_state_city_map)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5001)