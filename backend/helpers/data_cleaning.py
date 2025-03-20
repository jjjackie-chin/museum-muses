
def getDataset():
  import pandas as pd
  #import matplotlib.pyplot as plt
  import numpy as np

  #import data
  mus_rev_dict = pd.read_json("./data/review_quote_USonly.json", typ='series')
  mus_cat_dict = pd.read_json("./data/museum_categories_USonly.json", typ='series')
  # df = pd.read_csv("./data/tripadvisor_museum_USonly.csv")

  # resetting index
  mus_rev = mus_rev_dict.reset_index()
  mus_cat = mus_cat_dict.reset_index()

  # naming col
  mus_rev.columns = ['MuseumName', 'Reviews']
  mus_cat.columns = ['MuseumName', 'Categories']

  # join
  merged = pd.merge(mus_cat, mus_rev, on='MuseumName', how='inner')
  return merged

  # # explore data
  # mus_rev_keys = mus_rev_dict.keys() #1007 museums, 10-15 review quotes for each



# testlist = df["Gettysburg Heritage Center"]
# df.shape
# df.columns
# df.head()


## retrieving museums from a given category
def getMuseums(user_input_cat):
  """
  Takes in  the category that the user pre selected and returns a list of all 
  museum names that are in that category

  Parameters:
    user_input_cat: str of the category that user selects (ex: "Art Museums")
    dataset: DataFrame of MuseumName, Categories, Reviews

  Returns:
    matching: list[str] list of museum names in that category

  """
  dataset = getDataset()
  cat_exploded = dataset.explode('Categories')
  matching = cat_exploded[cat_exploded['Categories'] == user_input_cat]['MuseumName'].tolist()

  return matching

