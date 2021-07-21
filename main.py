from prediction.content_classification import test_model, train_mnb, train_sgd, train_nn
import click

# Filepaths for train and test datasets
TRAIN_DATASET_PATH = "data/news_dataset.json"
TEST_DATASET_PATH = "data/post_data.csv"

@click.group()
def main():
    pass

@main.command(help="Command to predict categories based on a model")
@click.option('-m', '--model', required=True, help="Path to saved model")
def predict(model: str):
    test_model(TEST_DATASET_PATH, model)

@main.group(help="Training Commands")
def train():
    pass

@train.command(help="Trains a model using sklearn MultinomialNB")
@click.option('-n', '--name', default='multinomialnb', help='what to name trained model')
def mnb(name: str):
    train_mnb(TRAIN_DATASET_PATH, name)

@train.command(help="Trains a model using sklearn SGD")
@click.option('-n', '--name', default='sgd', help='what to name trained model')
def sgd(name: str):
    train_sgd(TRAIN_DATASET_PATH, name)

@train.command(help="Trains a model using sklearn MLP")
@click.option('-n', '--name', default='nn', help='what to name trained model')
def nn(name: str):
    train_nn(TRAIN_DATASET_PATH, name)

if __name__ == "__main__":
    main()