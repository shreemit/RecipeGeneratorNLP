import re
import pandas as pd
import re
from sklearn.model_selection import train_test_split

import json

with open('layer1.json') as json_file:
    data = json.load(json_file)

train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)


def get_instructions(recipe):
    instructions = recipe['instructions']
    instr_str = "recipe: "
    for instr in instructions:
        recipe = re.sub(r"\s", " ", instr['text'])
        instr_str += recipe.lower().strip()
    return instr_str


def build_text_files(recipes, dest_path):
    f = open(dest_path, 'w')
    in_recipe = ''
    for recipe in recipes:
        instr = get_instructions(recipe)
        text = re.sub(r"\s", " ", instr)
        in_recipe += text
    f.write(in_recipe)


train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

build_text_files(train_data, train_path)
build_text_files(test_data, test_path)
