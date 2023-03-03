import re
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import opinion_lexicon
import re
import json
import string

with open('layer1.json') as json_file:
    data = json.load(json_file)

train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)


# def get_instructions(recipe):
#     instructions = recipe['instructions']
#     instr_str = "recipe: "
#     for instr in instructions:
#         recipe = re.sub(r"\s", " ", instr['text'])
#         instr_str += recipe.lower().strip()
#     return instr_str
def get_ingredients(recipe):
    ingredients = recipe['ingredients']
    measurements = ['oz', 'ounces', 'ounce', 'cups', 'cup', 'teaspoon', '/', 'to', 'lbs', 'lb', 'tsp', 'jar']
    item_list = []
    for item in ingredients:
        item = item['text']
        item = item.lower()
        item = item.translate(str.maketrans('', '', string.punctuation))
        #     item = item.split('(')[0].strip() # remove items in brackets
        item = re.sub(r'\s*\([^)]*\)', '', item)
        item = ''.join([char for char in item if not char.isdigit()])  # remove numbers
        for measurement in measurements:
            item = item.replace(measurement, '').strip()  # remove measurements
        item.strip()
        if item:
            item_list.append(item)
    ingredient_list = '<SOI> ' + ' <ISEP> '.join(item_list) + ' <EOI>'
    return ingredient_list


def get_instructions(recipe):
    instructions = recipe['instructions']
    instr_str = "<SOR> "
    for instr in instructions:
        recipe = re.sub(r"\s", " ", instr['text'])
        item = item.translate(str.maketrans('', '', string.punctuation))
        instr_str += recipe.lower().strip()
    instr_str += ' <EOR>'
    return instr_str


def build_df(recipes):
    in_recipe = []
    for recipe in recipes:
        ingred = get_ingredients(recipe)
        instr = get_instructions(recipe)
        text = {"text": ingred + instr}
        in_recipe.append(text)
    df = pd.DataFrame(in_recipe)
    return df


train_df = build_df(train_data)
test_df = build_df(test_data)

train_df.to_csv('train_dataframe.csv')
test_df.to_csv('test_dataframe.csv')
print(train_df.head())
