
---

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

1. **Setup and Import dependancies.**

2. **Load and Prepare Dataset:**

3. **Model Selection:**
    - Select **BERT-base-uncased**; A good starting point for general NER tasks. Consider factors like model size, training data, and computational power. 
    - Load the pre-trained BERT model and add a classification head on top. 
    - Add **Classification Head**; This layer will output predictions for each token in the sequence, indicating the corresponding entity type.

4. **Fine-Tuning:** 
    - **Learning Rate:** I used **1e-4** as recommended in BERT documentation.
    - **Batch Size:** I used 4 to save memory (RAM).
    - **Epochs:** I used only one epoch because of the shortage of resources.
    - Iterate through the dataset, feed the prepared data to the BERT model, calculate the loss, and update the model's weights using the optimizer. 

5. **Evaluation:**

Evaluate the performance of your fine-tuned BERT model using appropriate metrics for NER such as Precision, Recall, and F1-score:

6. **Making Predictions:**

    - **Load the Fine-Tuned Model:** Load the saved state of your fine-tuned BERT model.
    - **Input Text:** Provide new text data that you want to analyze for named entities.
    - **Run Inference:** Pass the new text through the loaded model.
    - **Generate Predictions:** Obtain predictions from the model, indicating the entity types for each token. 

### 7. Overall Conclusion

The three NER methods tested with a 1.5 million data point dataset reveal a trend of increasing performance as model complexity increases. 

* **Baseline (Word Embeddings & Logistic Regression):** Despite its simplicity, the baseline achieved an accuracy of 88%, highlighting the effectiveness of pre-trained word embeddings for capturing semantic meaning. However, its performance on individual tags was inconsistent, indicating limitations in capturing context for complex NER tasks. This model Con NOT capture the context between given phrases.

* **LSTM from Scratch:** This model, trained from scratch, showed improvement over the baseline, likely due to its ability to capture long-range dependencies within sentences. It achieved a significant drop in loss throughout training, suggesting better learning compared to the baseline. This highlights the benefit of recurrent networks for sequential data like text. Despite having only one LSTM layer, it takes too long to complete tasks (8.3 hours) on the CPU.

* **Fine-tuned BERT:** The fine-tuned BERT model achieved the highest F1-score, demonstrating its powerful capabilities in capturing complex language patterns. Its performance on this dataset is comparable to state-of-the-art models on smaller datasets. However, the limited training epochs due to computational constraints might have hindered its full potential.

**Comparison to Sentiment Analysis Benchmarks:**

Directly comparing these results to sentiment analysis benchmarks is challenging due to the different task nature. However, some observations can be made:

* **Dataset Size:** The 1.5 million data points used in this NER task are significantly larger than datasets typically used in sentiment analysis (often in the range of tens of thousands). This larger dataset size likely contributed to the higher performance achieved by the more complex models.
* **Model Complexity:** The use of deep learning models like LSTMs and BERT in NER is common, similar to sentiment analysis. However, in sentiment analysis, simpler models like Naive Bayes or SVM often perform surprisingly well, especially with well-engineered features. 
* **Fine-tuning:** The fine-tuning approach used with BERT is highly effective in sentiment analysis, achieving state-of-the-art results on various benchmarks. The similar success in NER underscores the transferability of BERT's knowledge across diverse NLP tasks.

**Future Directions:**

* **Hyperparameter Optimization:** Further exploration of hyperparameters for each model could lead to improved performance. 
* **Larger Models:** Exploring larger BERT models (e.g., BERT-large-uncased) or even specialized NER models like BERT-NER could yield even better results.


**Benchmarks in NER**

Here are some existing benchmarks in NER and their key characteristics:
| Dataset | Dataset Description | Significance | Key Metrics | State-of-the-art Performance |
|---|---|---|---|---|
| CoNLL-2003 Shared Task | News articles annotated with four entity types: PERSON, LOCATION, ORGANIZATION, MISC. | Widely used benchmark for NER, providing a standard dataset for comparing models. | F1 score, Precision, Recall | BERT-based models (around 90% F1 score) |
| GENIA Corpus | Biomedical literature annotated with gene and protein names. | Important for NER in the domain of biomedicine. | F1 score, Precision, Recall | Specialized bio-NER models achieving high F1 scores (above 90%) |
| OntoNotes 5.0 | Diverse text genres (news, blogs, web text) with 18 entity types. | Covers a broader range of entity types and diverse text styles. | F1 score, Precision, Recall | BERT-based models, achieving F1 scores above 90% |
| WNUT 2017 NER Task | Twitter data with 10 entity types. | Focuses on NER in social media text, addressing the challenges of informal language and noise. | F1 score, Precision, Recall | Models achieving F1 scores in the 80s, reflecting the complexity of social media text |
| ACE 2005 | English broadcast news transcripts with 7 entity types. | Provides a benchmark for NER in news transcripts, considering temporal information and complex entities. | F1 score, Precision, Recall | Specialized models for news transcript NER, achieving high F1 scores | 

### 8. Reflection Questions

1. **What was the biggest challenge you faced in implementing Named Entity Recognition?** 
Here are the key challenges faced in implementing Named Entity Recognition (NER), summarized in bullet points:

    * Dealing with large datasets.
    * Data biases toeards `O` tag.
    * Complex models like LSTMs and BERT need more code and data processing.
    * Finding optimal hyperparameters for these models is time-consuming and computationally expensive.


2. **What insights did you gain about NLP and NER through this project?** 

Here are some key insights I gained about NLP and NER through this project:

* The performance of NLP models is heavily reliant on the quality and quantity of training data. Even the most sophisticated models struggle without sufficient and diverse training data.

* Pre-trained models like BERT are powerful. They provide a strong foundation for various downstream tasks and often achieve impressive performance with minimal fine-tuning.

*  Understanding context is crucial for many NLP tasks, including NER. Models need to go beyond simply recognizing individual words and instead learn how words relate to each other within a sentence and the broader context of the document.

* Simple models like Logistic Regression can provide a baseline, but more complex models like LSTMs and BERT are better equipped to handle the complexities of NER, especially with large datasets.

