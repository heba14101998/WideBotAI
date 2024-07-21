
# Named Entity Recognition (NER) for Text Analysis
---

This document presents the findings of a Named Entity Recognition (NER) project implemented using Python and various libraries. The goal is to identify and classify entities within text data, contributing to a deeper understanding of the text content.

### 1. Introduction

Named Entity Recognition (NER) is a crucial task in natural language processing (NLP), focusing on identifying and categorizing named entities within text. These entities can be people, locations, organizations, dates, or other concepts. By performing NER, we can extract valuable information from text and gain insights into the context.

This project aims to develop and evaluate a NER system for text analysis using the Kaggle NER Dataset. The system is designed to identify and classify entities in a given text corpus, providing a foundation for further NLP applications.

### 2. Data Description

The dataset used in this project is the [Kaggle NER Dataset](https://www.kaggle.com/datasets/namanj27/ner-dataset), a collection of text data annotated with entity types. It consists of a variety of texts, including news articles, social media posts, and other forms of written communication. 
The dataset with 1M x 4 dimensions contains columns = ['# Sentence', 'Word', 'POS', 'Tag'] and is grouped by Sentence #.

##### Word: 
This column contains English dictionary words form the sentence it is taken from.

##### POS: 
This column contains the Parts of speech tag

##### Tag: 
This column contains:

* **Person:** Names of individuals. Eddy Bonte, e,g,. President Obama
* **Location:** Names of geographical entities. e,g,. Murray River, Mount Everest.
* **Organization:** Names of organizations. e,g,. Georgia-Pacific Corp., WHO
* **Date:** Dates and time expressions. e,g,. June, 2008-06-29.
* **Time:** Time expressions. e,g,. two fifty a m, 1:30 p.m.
* **Money:** Monetary values. e,g,. 175 million Canadian Dollars, GBP 10.40.
* **Percentage:** Percentages. e,g,. twenty pct, 18.75 %.
* **GPE:** Geopolitical entities. e,g,. South East Asia, Midlothian.
* **FACILITY:** Legal documents, acts, cases, etc. Washington Monument, Stonehenge

### 3. Tools and Libraries

* Python 3.9
* ipykernel==6.29.5
* opendatasets==0.1.22
* numpy==1.26.0
* pandas==2.2.2
* scipy==1.10.1
* matplotlib==3.6.0
* seaborn==0.12.0
* nltk==3.8.1
* spacy==3.7.5
* gensim==4.3.0
* scikit-learn==1.5.1
* scikit-multilearn==0.2.0
* torch==2.3.1
* simpletransformers==0.70.1

### 4. Baseline Experiments

**Goal:** To establish a baseline for model performance using a simple approach, leveraging word embeddings and a traditional machine learning model.

**Methodology:**

1. **Text Representation:** Text Representation using word-to-vec mapping to represent each word seperatly to its corresponding nummerical vector. 
   - Load the pre-trained GloVe embeddings (`glove-twitter-25`) using gensim.downloader.
   - Create a function `get_vector` to handle Out-of-Vocabulary (OOV) words by generating random vectors.
   - Apply this function to the `Word` column to create the `WordVector` column.

2. **Data Splitting:**
   - Split the data into training and testing sets (80% for training, 20% for testing).

3. **Model Training:**
   - A Logistic Regression model is used.
   - Train the model on the vectorized training data.

4. **Model Evaluation:** Using my utils module [utils.py](https://github.com/heba14101998/WideBotAI/blob/main/utils.py)
   - Calculate the accuracy of each model on the test (evaluation) data.
   - Plot the confusion matrices using Seaborn to visualize the model's predictions.
   - Print classification reports to get detailed performance metrics for each tag class.

5. **Make Prediction:** Make predictions on new unseen words using the trained models.


### 5. Advanced Experiments: LSTM

**Goal:** To improve upon the baseline performance by leveraging the power of recurrent neural networks (RNNs) specifically, Long Short-Term Memory (LSTM) networks. 

**Methodology:**

1. **Import dependancies and load dataset.**

2. **Data Preparation:** Preprocess the text data, including tokenization, padding, and encoding words and tags.
    -  Define a custom dataset class to load and prepare your data. It should inherit from `torch.utils.data.Dataset` and implement `__len__` and `__getitem__` methods.
    -  Create a DataLoader to handle batching and shuffling of data.

3. **Data Splitting**: use pytorch random split to split the data into train and test sets.
4. **Model Architecture:** Define the NER model architecture using LSTMs and configure the hyperparameters like the number of LSTM units and embedding dimensionality.
    - Create a class `LSTMModel` that inherits from `nn.Module`. Define the layers (LSTM and fully connected) and the forward pass logic.
    - Set the device (`device`) to "cuda" for GPU usage if available or "cpu" otherwise.
    - Define hyperparameters like input size, hidden size, number of layers, learning rate, batch size, and number of epochs.
    - Create an instance of your `LSTMModel` and move it to the device.
    - Choose an appropriate loss function (e.g., `nn.CrossEntropyLoss` for classification) and an optimizer (e.g., `optim.Adam`).

5. **Model Training:** Train the model on the prepared data, monitoring the performance through loss and accuracy metrics.
    - Iterate through epochs and batches.
    - Perform forward and backward passes.
    - Calculate the loss.
    - Update model weights using the optimizer.
    - Save the trained model's state dictionary using `torch.save`.

6. **Make Prediction:**
    - Load the saved model state dictionary for inference.
    - Use the loaded model to make predictions on new unseen data.

### 6. Advanced Experiment: BERT 

**Goal:** Enhance performance by utilizing the power of pre-trained language models, specifically BERT (Bidirectional Encoder Representations from Transformers).

<!-- 1. **Setup and Import dependancies.**

2. **Load and Prepare Dataset:**

2. **Model Selection:**
Select **BERT-base-uncased**; A good starting point for general NER tasks. Consider factors like model size, training data, and computational power. 

3. **Fine-Tuning BERT for NER:**

- **Initialize the BERT Model:** Load the pre-trained BERT model and add a classification head on top. 
    * **Classification Head:** This layer will output predictions for each token in the sequence, indicating the corresponding entity type.

- **Fine-Tuning:** 
    * **Objective:** Train the BERT model by fine-tuning its weights to perform NER.
    * **Loss Function:** Use a suitable loss function for classification tasks, like `CrossEntropyLoss`.
    * **Optimizer:** Choose an optimizer to update the model weights during training (e.g., AdamW).
    * **Training Process:** Iterate through the dataset, feed the prepared data to the BERT model, calculate the loss, and update the model's weights using the optimizer.   


**4. Training Configuration:**

* **Hyperparameters:**  Experiment with these hyperparameters to optimize performance:
    * **Learning Rate:** Determines the step size for weight updates.
    * **Batch Size:** Number of samples processed at once during training.
    * **Epochs:** Number of times the entire dataset is passed through the model during training.
    * **Warmup Steps:**  A gradual increase in learning rate at the beginning of training to help the model converge faster.

**5. Evaluation:**

* **Metrics:** Evaluate the performance of your fine-tuned BERT model using appropriate metrics for NER:
    * **F1 Score:** A balanced measure of Precision and Recall, often used as the primary metric.
    * **Precision:**  Accuracy of the model's predictions.
    * **Recall:**  Ability of the model to correctly identify all entities.

**6. Making Predictions:**

* **Load the Fine-Tuned Model:** Load the saved state of your fine-tuned BERT model.
* **Input Text:** Provide new text data that you want to analyze for named entities.
* **Run Inference:** Pass the new text through the loaded model.
* **Generate Predictions:** Obtain predictions from the model, indicating the entity types for each token. -->
<!-- 

3. **Data Splitting**: 

4. **Model Architecture:** 

5. **Model Training:** 
6. **Make Prediction:**
    - Load the saved model state dictionary for inference.
    - Use the loaded model to make predictions on new unseen data. -->

### 7. Overall Conclusion

The experiments conducted in this project will demonstrate the effectiveness of different approaches to Named Entity Recognition. Comparing the performance of the baseline, LSTM, and BERT models will highlight the advantages of using more advanced techniques like deep learning and pre-trained models.

<!-- مميزات وعيوب كل طريقه -->

<!-- جدول مقارنه هنا  -->
### Benchmarks in NER

Here are some existing benchmarks in NER and their key characteristics:

**1. CoNLL-2003 Shared Task:**

* **Dataset:** News articles annotated with four entity types: PERSON, LOCATION, ORGANIZATION, MISC. 
* **Significance:** Widely used benchmark for NER, providing a standard dataset for comparing models.
* **Key Metrics:** F1 score, Precision, Recall.
* **State-of-the-art Performance:** BERT-based models (around 90% F1 score).

**2. GENIA Corpus:**

* **Dataset:** Biomedical literature annotated with gene and protein names.
* **Significance:** Important for NER in the domain of biomedicine.
* **Key Metrics:** F1 score, Precision, Recall.
* **State-of-the-art Performance:** Specialized bio-NER models achieving high F1 scores (above 90%).

**3. OntoNotes 5.0:**

* **Dataset:** Diverse text genres (news, blogs, web text) with 18 entity types.
* **Significance:** Covers a broader range of entity types and diverse text styles.
* **Key Metrics:** F1 score, Precision, Recall.
* **State-of-the-art Performance:** BERT-based models, achieving F1 scores above 90%.

**4. WNUT 2017 NER Task:**

* **Dataset:** Twitter data with 10 entity types.
* **Significance:** Focuses on NER in social media text, addressing the challenges of informal language and noise.
* **Key Metrics:** F1 score, Precision, Recall.
* **State-of-the-art Performance:** Models achieving F1 scores in the 80s, reflecting the complexity of social media text.

**5. ACE 2005:**

* **Dataset:** English broadcast news transcripts with 7 entity types.
* **Significance:** Provides a benchmark for NER in news transcripts, considering temporal information and complex entities.
* **Key Metrics:** F1 score, Precision, Recall.
* **State-of-the-art Performance:** Specialized models for news transcript NER, achieving high F1 scores.

#### Compare your results to existing benchmarks in NER:
<!-- قارن هنا  -->

### 8. Reflection Questions

1. **What was the biggest challenge you faced in implementing Named Entity Recognition?** 

2. **What insights did you gain about NLP and NER through this project?** 

This project will provide valuable insights into the capabilities and limitations of various NLP techniques for NER. It will also shed light on the importance of data preparation, model selection, and hyperparameter tuning for achieving optimal results. 
