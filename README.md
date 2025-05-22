# NLP Sentiment Analysis BiLSTM

This repository contains the code for a Natural Language Processing (NLP) project that implements a sentiment analysis model using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. The model is designed to classify text data according to sentiment, leveraging deep learning techniques and pre-trained word embeddings.

## Project Overview

- **Dataset:** Uses a publicly available text dataset (for example, IMDb movie reviews) to train and evaluate the sentiment classification model.
- **Preprocessing:** Text data undergoes cleaning, tokenisation, removal of stopwords, and is transformed into numerical representations using pre-trained embeddings such as GloVe or Word2Vec.
- **Model:** The core model is a BiLSTM neural network composed of embedding layers, bidirectional LSTM layers, and fully connected dense layers for sentiment prediction.
- **Evaluation:** The model is evaluated using metrics including accuracy, precision, recall, and F1-score. Visualisations such as learning curves and confusion matrices are provided to aid understanding of model performance.
- **Environment:** The code is fully runnable on Google Colab or any Python environment supporting TensorFlow (or PyTorch), with dependencies managed via `requirements.txt`.


## Features

- Comprehensive text preprocessing pipeline: cleaning, tokenisation, and stopwords removal
- Utilisation of pre-trained word embeddings (GloVe/Word2Vec) for feature representation
- Bidirectional LSTM model architecture with configurable hyperparameters
- Training and validation loops with detailed loss and accuracy tracking
- Model evaluation with quantitative metrics and visualisations
- Modular and well-documented codebase designed for ease of use and reproducibility

## Getting Started

### Prerequisites

- Python 3.7 or later  
- TensorFlow (or PyTorch, depending on implementation)  
- Common Python libraries: numpy, pandas, matplotlib, seaborn  
- NLP libraries: NLTK or SpaCy for preprocessing  
- Jupyter Notebook or Google Colab for interactive execution

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/nlp-sentiment-bilstm.git
cd nlp-sentiment-bilstm
```

### Install the required Python packages:
pip install -r requirements.txt

## Usage

Open the Jupyter Notebook `sentiment_analysis_bilstm.ipynb` in Jupyter or Google Colab.

The notebook is organised in the following main steps:

1. **Data Loading and Preprocessing:**  
   Load the text dataset, clean the data, tokenize the text, remove stopwords, and prepare input sequences.

2. **Embedding Layer Preparation:**  
   Load pre-trained word embeddings (such as GloVe or Word2Vec) and map them to the dataset vocabulary.

3. **Model Building:**  
   Define the BiLSTM model architecture including embedding, bidirectional LSTM, dropout, and dense layers.

4. **Training:**  
   Train the model with training data, including validation, and track metrics like loss and accuracy.

5. **Evaluation:**  
   Evaluate the model using accuracy, precision, recall, and F1-score. Visualise performance with learning curves and confusion matrices.

6. **Inference:**  
   Test the model with sample text inputs to predict sentiment.

Each cell is well commented to explain the code functionality.

## Project Structure

- The notebook contains all the code from data preprocessing to model evaluation.
- Data and embeddings folders may be optionally included or downloaded dynamically.
- `requirements.txt` ensures reproducible environment setup.

## Notes

- This project is designed for educational purposes and can be extended for other NLP tasks.
- Ensure you have a stable internet connection if running on Google Colab to download datasets and embeddings.
- For any questions or issues, feel free to open an issue in this repository.
