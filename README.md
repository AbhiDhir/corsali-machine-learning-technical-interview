# Corsali Machine Learning Technical Interview

This is our project-based technical interview for machine learning engineers. It gives us a chance to see your abilities and how you approach problems. It is designed to give you creative freedom as you develop a solution, and consists of a model implementation quesion as well as non-coding questions below. Please complete both the coding and non-coding questions.

Please download the repo as a zip file for your submission.

## Coding questions

Please build a content classification model that takes in a news headline as an input and predicts the category as an output. There are two datasets included: `news_dataset.json` which contains 200,000 headlines covering 41 categories, and `post_data.csv` which contains 400 sample posts on which the model will be used. The content classification model will predict one of the 41 categories from the news dataset. The goal is to create a performant and accurate model. You are welcome to use an off-the-shelf model, a pre-trained model, or a custom trained model, or any combination of them. Feel free to use any packages/tools/etc to build a great model.

Please write the code for the model in content_classification.py, and call it from main.py. Please report the accuracy of the model as evaluated on the test dataset, which you can choose to split any way you'd like.

***

## Usage

### Requirements/Installation

```
python3 -m pip install -r requirements.txt
```

Additionally one may have to download the `nltk` library's wordnet via: 
```
import nltk
nltk.download('wordnet')
```

### Functionality

This tool is a command line interface using the `click` library that can be run with the basic command `python3 main.py`

There are two subcommands, `train` and `predict`. Running `python3 main.py train` will show a list of models that can be trained, with an optional flag of `-n` that can be used to name the model. Running `python3 main.py predict` requires the `-m` flag with a path to a saved model in order for the command to execute. This will use the saved model to create a prediciton on the `post_data.csv` dataset.

### Directory Structure

- `data/` holds the training and test data, along with the preprocessed/cleaned versions of them
- `models/` holds saved models
- `prediction/` holds training/prediction code
- `predictions/` holds csv files of predictions for each trained model

***
## Results

First, all of the data was preprocessed in order to remove stop words and punctuation, make them all lowercase, and lemmatize the words. Then each model was run using a variety of paremeters via sklearn's `GridSearchCV` with cross validation. The overall accuracies were pretty poor as they were only around 50%. However, in this project I was optimizing for speed rather than accuracy as I wanted each model to finish training on the full data in a few minutes. Some experimentation was also done with using TruncatedSVD as a PCA substitute (for sparse data) on the data in order to reduce dimensionality and speed up the training time. However, this reduced the accuracy significantly and was thus not used.

### Accuracies

- MultinomialNB: 50.60%
- SGD (Full Dataset): 51.13%
    - SGD (10,000 datapoints (probably overfit)): 68.32%
- Simple Neural Network (MLP): 39.01%

***

## Non-coding questions

Please answer the non-coding questions below by editing this readme

1. You are given a dataset of 1500 company descriptions and must classify them into 6 categories. You train a logistic regression, but it only has 60% accuracy. Your task is to identify which types of data are underrepresented in the dataset in order to prioritize

It seems that this dataset is imbalanced and that the underrepresented categories in the datset may never be predicted. In this case not only is accuracy important, but so are measures such as precision, recall, and the resulting F-measure of each category. In order to prioritize certain categories (minimize false negatives) we can start off the logistic regression with different weights per category using the Weighted Maximum Liklihood method. This will use the weighting to influence the sampling proportions for the logistic regression.

2. You are choosing the best way to represent a text company description as an input to a model. Describe the trade offs between bag-of-words, word embeddings, and any other representation you may choose.

Bag-of-words is a simple model that can taken in a large corpus. However it has a very large dimensionality and does not take into account the order of the words. Word embeddings are more complex where words are mapped to vecotrs of real numbers. The vectors take into account probability distributions for words appearing before and after each other. This would be a better method to use if the dataset is much larger and the ordering of words gives more meaning. Lastly, there are more complex embeddings such as Google's BERT which looks at the entire sentence befroe assigning a vector to each word, so words can have different vectors based on context. This is more robust than simple word embeddings, but also would take longer in preprocessing and may perform worse than even bag-of-words on very small datasets.

3. You decide to use a recurrent neural network for the text classification task instead. You implement the model in PyTorch and find that it's very accurate, but that the model evaluation is too slow. How would you accelerate the model prediction time?

There are a few different things I could try. First I would check the pytorch tuning guide to make sure all functon calls are optimized. Next I would try to save the model to ONNX format and try to convert it to a tflite model which uses quantization to run faster. I would do these steps first in an effort to preserve accuracy. If they do not work, I would then try simplifying the model by reducing the size of the layers or number of layers in the network with a minimal loss in accuracy.

***

## Submission

Once complete, please email a zip file to: jobs@corsali.com or your current contact

*** 

## Sources

- https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958
- https://www.kaggle.com/kinguistics/classifying-news-headlines-with-scikit-learn
- https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
- https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
