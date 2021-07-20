# Corsali Machine Learning Technical Interview

This is our project-based technical interview for machine learning engineers. It gives us a chance to see your abilities and how you approach problems. It is designed to give you creative freedom as you develop a solution, and consists of a model implementation quesion as well as non-coding questions below. Please complete both the coding and non-coding questions.

Please download the repo as a zip file for your submission.

## Coding questions

Please build a content classification model that takes in a news headline as an input and predicts the category as an output. There are two datasets included: `news_dataset.json` which contains 200,000 headlines covering 41 categories, and `post_data.csv` which contains 400 sample posts on which the model will be used. The content classification model will predict one of the 41 categories from the news dataset. The goal is to create a performant and accurate model. You are welcome to use an off-the-shelf model, a pre-trained model, or a custom trained model, or any combination of them. Feel free to use any packages/tools/etc to build a great model.

Please write the code for the model in content_classification.py, and call it from main.py. Please report the accuracy of the model as evaluated on the test dataset, which you can choose to split any way you'd like.

## Non-coding questions

Please answer the non-coding questions below by editing this readme

1. You are given a dataset of 1500 company descriptions and must classify them into 6 categories. You train a logistic regression, but it only has 60% accuracy. Your task is to identify which types of data are underrepresented in the dataset in order to prioritize

_Type your answer to question 1 here_

2. You are choosing the best way to represent a text company description as an input to a model. Describe the trade offs between bag-of-words, word embeddings, and any other representation you may choose.

_Type your answer to question 2 here_

3. You decide to use a recurrent neural network for the text classification task instead. You implement the model in PyTorch and find that it's very accuracte, but that the model evaluation is too slow. How would you accelerate the model prediction time?

_Type your answer to question 3 here_

## Submission

Once complete, please email a zip file to: jobs@corsali.com or your current contact
