# Sentiment Analysis with GloVe Embeddings

This project demonstrates **sentiment classification** on short text reviews using **deep learning (Keras/TensorFlow)** and **pre-trained GloVe word embeddings**.  
It uses datasets from Yelp, Amazon, and IMDB to train models that can classify text as **positive (1)** or **negative (0)**.

---

## 📊 Dataset

The dataset used comes from the [UCI Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences).  
It contains **sentences labeled with sentiment** across three domains:

- **Yelp** (yelp_labelled.txt)  
- **Amazon** (amazon_cells_labelled.txt)  
- **IMDB** (imdb_labelled.txt)  

Each file contains 1000 sentences with binary labels.

---

## 🔧 Approach

1. **Preprocessing**
   - Tokenize text with `Tokenizer` (limit to 5000 words).
   - Pad sequences to a fixed length (`maxlen=100`).
   - Train-test split: 75% training, 25% testing.

2. **Word Embeddings**
   - Pre-trained **GloVe (glove.6B.50d.txt)** embeddings.
   - Created embedding matrix mapping each token to a 50-dimensional vector.
   - Compared:
     - **Frozen embeddings** (non-trainable).
     - **Fine-tuned embeddings** (trainable).

3. **Model Architecture**
   - Embedding Layer (initialized with GloVe vectors).  
   - GlobalMaxPooling1D (dimensionality reduction).  
   - Dense Layer with ReLU activation.  
   - Output Layer (sigmoid) for binary classification.  

4. **Training**
   - Optimizer: Adam  
   - Loss: Binary Crossentropy  
   - Metrics: Accuracy  
   - Epochs: 10–50  
   - Batch Size: 10  

---

## 📈 Results

- **Frozen embeddings**:  
  - Training Accuracy ≈ **68%**  
  - Testing Accuracy ≈ **65%**

- **Fine-tuned embeddings**:  
  - Training Accuracy ≈ **100%** 
  - Testing Accuracy ≈ **81%**  

These results are reasonable given the **small dataset size** and **simple model architecture**.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/adityabansal28/SentimentAnalysis_GloVe.git
   cd SentimentAnalysis_GloVe
