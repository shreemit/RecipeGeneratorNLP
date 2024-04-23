# Recipe Generator NLP Project

This project aims to develop a Natural Language Processing (NLP) model capable of generating recipes based on given ingredients. The system leverages the power of transformer models, specifically T5, to understand and generate textual data. The project is divided into several key components, each playing a crucial role in the overall functionality of the recipe generation system.

## Components

### 1. Data Preparation

The foundation of the project lies in the preparation of the dataset. The `generate_dataset.py` script is responsible for processing raw recipe data into a structured format suitable for training the model. It performs several preprocessing steps, including:

- Splitting the dataset into training and testing sets.
- Extracting and cleaning ingredient and instruction data from recipes.
- Removing unnecessary characters, numbers, and measurements from ingredient names.
- Tokenizing and formatting instructions for the model.

### 2. Tokenization and Dataset Creation

The `train.py` script handles the tokenization of the dataset and the creation of the training and testing datasets. It uses the Hugging Face `transformers` library to tokenize the text data, adding special tokens for the start and end of sentences, as well as separators for ingredients and instructions. The script also groups texts into blocks of a specified size, ensuring that the model can process the data efficiently.

### 3. Model Training

The core of the project is the training of the T5 model on the prepared dataset. The `train.py` script sets up the training environment, including the model, tokenizer, and training arguments. It uses the Hugging Face `Trainer` class to manage the training process, including saving the model and evaluating its performance.

### 4. Testing and Evaluation

The `test.py` script demonstrates how to use the trained model to generate recipes. It loads the trained model and tokenizer, then uses the `pipeline` function from the `transformers` library to create a text generation pipeline. This pipeline can be used to generate recipes based on given ingredients, showcasing the model's ability to understand and generate textual data.

### 5. Data Analysis

The `generate_dataset.py` script and the `DataAnalysis` class within it provide tools for analyzing the dataset. This includes visualizing the distribution of ingredient and instruction lengths, which can help in understanding the data and identifying potential issues or areas for improvement.

## Getting Started

To get started with the project, follow these steps:

1. Ensure you have Python 3.6 or later installed.
2. Install the required libraries by running `pip install -r requirements.txt`.
3. Run the `generate_dataset.py` script to prepare the dataset.
4. Train the model by running `main.py`.
5. Test the model by running `test.py`.

## Training the Model

To train the model, follow these steps:

1. Prepare the dataset by running `generate_dataset.py`. This will create `train_df.csv` and `test_df.csv` files.
2. Train the model by executing `main.py`. This script initializes the T5 model, sets up the training and validation datasets, and trains the model.
3. The trained model will be saved in the `trained_models` directory.

## Testing the Model

To test the model, follow these steps:

1. Ensure you have a trained model saved in the `trained_models` directory.
2. Run `test.py` to test the model. This script loads the trained model, processes the test dataset, and generates predictions.
3. The script calculates and prints BLEU, ROUGE, and BERT scores for the generated predictions.

## Further Possible Improvements

- **Improve Data Preprocessing**: Enhance the data preprocessing steps in `generate_dataset.py` to handle more edge cases and improve the quality of the dataset.
- **Experiment with Different Models**: Try training other transformer models like GPT-3 or BART to see if they offer better performance.
- **Hyperparameter Tuning**: Experiment with different hyperparameters for the T5 model, such as learning rate, batch size, and number of training epochs.
- **Incorporate More Data**: If available, incorporate more recipe data to improve the model's ability to generalize.
- **Enhance Evaluation Metrics**: Consider using additional evaluation metrics or incorporating user feedback to further refine the model's performance.

## Contributing

Contributions to the project are welcome. Please feel free to submit pull requests or open issues on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.