# Recipe Generator NLP Project

This project aims to develop a Natural Language Processing (NLP) model capable of generating recipes based on given ingredients. The system leverages the power of transformer models, specifically GPT-2, to understand and generate textual data. The project is divided into several key components, each playing a crucial role in the overall functionality of the recipe generation system.

## Components

### 1. Data Preparation

The foundation of the project lies in the preparation of the dataset. The `generate_dataset.py` script is responsible for processing raw recipe data into a structured format suitable for training the model. It performs several preprocessing steps, including:

- Splitting the dataset into training and testing sets.
- Extracting and cleaning ingredient and instruction data from recipes.
- Removing unnecessary characters, numbers, and measurements from ingredient names.
- Tokenizing and formatting instructions for the model.

### 2. Tokenization and Dataset Creation

The `Training.py` script handles the tokenization of the dataset and the creation of the training and testing datasets. It uses the Hugging Face `transformers` library to tokenize the text data, adding special tokens for the start and end of sentences, as well as separators for ingredients and instructions. The script also groups texts into blocks of a specified size, ensuring that the model can process the data efficiently.

### 3. Model Training

The core of the project is the training of the GPT-2 model on the prepared dataset. The `Training.py` script sets up the training environment, including the model, tokenizer, and training arguments. It uses the Hugging Face `Trainer` class to manage the training process, including saving the model and evaluating its performance.

### 4. Testing and Evaluation

The `Testing.py` script demonstrates how to use the trained model to generate recipes. It loads the trained model and tokenizer, then uses the `pipeline` function from the `transformers` library to create a text generation pipeline. This pipeline can be used to generate recipes based on given ingredients, showcasing the model's ability to understand and generate textual data.

### 5. Data Analysis

The `RecipeTraining.ipynb` notebook and the `DataAnalysis` class within the `generate_dataset.py` script provide tools for analyzing the dataset. This includes visualizing the distribution of ingredient and instruction lengths, which can help in understanding the data and identifying potential issues or areas for improvement.

## Getting Started

To get started with the project, follow these steps:

1. Ensure you have Python 3.6 or later installed.
2. Install the required libraries by running `pip install -r requirements.txt`.
3. Run the `generate_dataset.py` script to prepare the dataset.
4. Train the model by running `Training.py`.
5. Test the model by running `Testing.py`.

## Contributing

Contributions to the project are welcome. Please feel free to submit pull requests or open issues on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.