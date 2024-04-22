import json
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class RecipeData:
    def __init__(self, filepath):
        with open(filepath) as json_file:
            self.data = json.load(json_file)
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.1, shuffle = True)

    def get_ingredients(self, recipe):    
        ingredients = recipe['ingredients']
        measurements = ['oz', 'ounces','ounce', 'cups', 'cup', 'teaspoon', '/', 'to', 'lbs','lb', 'tsp', 'jar']
        item_list = []
        for item in ingredients:
            item = item['text']
            item = item.lower()
            item = item.translate(str.maketrans('', '', string.punctuation))
            item = re.sub(r'\s*\([^)]*\)', '', item)
            item = ''.join([char for char in item if not char.isdigit()]) # remove numbers
            for measurement in measurements:
                item = item.replace(measurement, '').strip() # remove measurements
            item.strip()
            item = re.sub('\s\s+', ' ', item)
            if item:
                item_list.append(item)

        ingredient_list = ' </I> '.join(item_list)
        return ingredient_list

    def get_instructions(self, recipe):
        instructions = recipe['instructions']
        instr_str = ""
        for instr in instructions:
            recipe = re.sub(r"\s", " ", instr['text'])
            recipe = re.sub(r'[^\w\s]', ' ', recipe)
            recipe = re.sub('\s\s+', ' ', recipe)
            instr_str += " " + recipe.lower().strip() + "."
        return instr_str

    def build_text_files(self, recipes):
        in_recipes=[]
        for recipe in tqdm(recipes):
            ingredients = self.get_ingredients(recipe)
            instr = self.get_instructions(recipe)
            qna = { "title": recipe['title'],
                    "ingred": ingredients,
                    "instructions": instr }
            in_recipes.append(qna)
        return in_recipes

    def process_data(self):
        train_df = self.build_text_files(self.train_data)
        test_df = self.build_text_files(self.test_data)
        train_df = pd.DataFrame(train_df)
        test_df = pd.DataFrame(test_df)
        train_df.to_csv('train_df.csv')
        test_df.to_csv('test_df.csv')
        return train_df, test_df

class DataAnalysis:
    def __init__(self, df):
        self.df = df.copy()

    def get_distribution(self):
        x = self.df.copy()
        x['ingred_len'] = self.df['ingred'].apply(lambda x: len(x.split()))
        x['instr_len'] = self.df['instructions'].apply(lambda x: len(x.split()))

        # find the rows lying in the 25th and 75th percentile
        ingred_25 = x['ingred_len'].quantile(0.25)
        ingred_75 = x['ingred_len'].quantile(0.75)
        instr_25 = x['instr_len'].quantile(0.25)
        instr_75 = x['instr_len'].quantile(0.75)

        # find the rows lying in the 25th and 75th percentile
        x = x[(x['ingred_len'] > ingred_25) & (x['ingred_len'] < ingred_75) & (x['instr_len'] > instr_25) & (x['instr_len'] < instr_75)].reset_index(drop=True)

        plt.show()

        return x

if __name__ == "__main__":
    recipe_data = RecipeData('Recipe/recipe1M_layers/layer1.json')
    train_df, test_df = recipe_data.process_data()

    data_analysis = DataAnalysis(train_df)
    x = data_analysis.get_distribution()

    data_analysis_test = DataAnalysis(test_df)
    x_test = data_analysis_test.get_distribution()

    x.to_csv('train_df.csv')
    x_test.to_csv('test_df.csv')