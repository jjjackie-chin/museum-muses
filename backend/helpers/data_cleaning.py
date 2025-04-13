import pandas as pd
import numpy as np


## helper funtion to get city and state
def get_city_state(df):
  '''
  Modifies df by adding two new columns, 'City' and 'State'

  Parameters:
    df: DataFrame containing info of museums (with 'Address' column of the form: 123 Green St, Ithaca, NY 14850)

  Returns:
    df: modified DataFrame with 'City' and 'State' columns added
  '''
  address_parts = df['Address'].str.split(', ')
  df['City'] = address_parts.str[-2]
  df['State'] = address_parts.str[-1].str[:2]

  return df
  
def getDataset():
  mus_rev_dict = pd.read_json("./data/review_content_USonly.json", typ='series')
  mus_cat_dict = pd.read_json("./data/museum_categories_USonly.json", typ='series')
  mus_quote_dict = pd.read_json("./data/review_quote_USonly.json", typ='series')

  # resetting index
  mus_rev = mus_rev_dict.reset_index()
  mus_cat = mus_cat_dict.reset_index()
  mus_quote = mus_quote_dict.reset_index()

  # naming col
  mus_rev.columns = ['MuseumName', 'Reviews']
  mus_cat.columns = ['MuseumName', 'Categories']
  mus_quote.columns = ['MuseumName', 'Content']

  # join
  merged = pd.merge(mus_cat, mus_rev, on='MuseumName', how='inner')
  print(merged.shape)

  merged = pd.merge(merged, mus_quote, on='MuseumName', how='left')
  print(merged.shape)

  # add info from trip advisor (location, description, fee, rating (0-5))
  trip_advisor = pd.read_csv("./data/tripadvisor_museum_USonly.csv")
  trip_advisor = trip_advisor[['MuseumName', 'Address', 'Description', 'Fee', 'Rating']]
  trip_advisor = get_city_state(trip_advisor)

  merged = pd.merge(merged, trip_advisor, on='MuseumName', how='inner')
  merged['City-State'] = merged['City'] + ", " + merged['State']
  return merged


## retrieving museums from given categories
def filterCategory(user_input_cat):
  """
  Takes in a list of categories that the user pre selected and returns a list of all 
  museum names that are in those categories

  Parameters:
    user_input_cat: list[str] list of the categories that user selects (ex: ["Art Museums", "History Museums"]).
    
    If parameter is an empty list, then no filter is applied

  Returns:
    matching: list[str] list of museum names in that category

  """
  dataset = getDataset()
  if len(user_input_cat)==0:
    return dataset['MuseumName'].tolist()
  cat_exploded = dataset.explode('Categories')
  filter_mask = pd.Series([False] * len(dataset))
  
  for cat in cat_exploded:
      location_mask = dataset['City-State'].str.contains(cat, case=False, na=False)
      filter_mask = filter_mask | location_mask

  matching = dataset[filter_mask]['MuseumName'].tolist()
  # print("Filtering by category...")
  return matching


## retrieving museums from the given locations (city, state)
def filterLocation(locations):
  """
  Takes in the locations that the user pre selected and returns a list of all 
  museum names from those locations

  Parameters:
    locations: list[str] list of location of the form "city name, state abbreviation"

    If parameter is an empty list, then no filter is applied

  Returns:
    matching: list[str] list of museum names in that category

  """
  dataset = getDataset()
    
  if len(locations) == 0:
      return dataset['MuseumName'].tolist()
  
  filter_mask = pd.Series([False] * len(dataset))
  
  for location in locations:
      location_mask = dataset['City-State'].str.contains(location, case=False, na=False)
      filter_mask = filter_mask | location_mask
  
  matching = dataset[filter_mask]['MuseumName'].tolist()
  print("Filtering by location...")
  # print(matching)
  return matching
