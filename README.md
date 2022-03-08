# ML Competition Project : Text Sentiment Classification

## Abstract

This paper details our approachto address the « EPFL ML Text ClassificationChallenge », where the aim is to classify wether a tweet message used to contain a positive ’:)’or negative ’:(’ smiley. Different approach were used : preprocessing, various word representations, classical machine learning models, neural network architectures, and transformers architecture.

## Results

The best result was obtained **finetuning a BERT** pre-trained model, giving an accuracy of **0.905** on AIcrowd where we were ranked **3th** among all the groups.

## Team members
 
 Adam Chellaoui  : adam.chellaoui@epfl.ch
 
 Douglas Bouchet : douglas.bouchet@epfl.ch
 
 Marina Rapellini : marina.rapellini@epfl.ch
 
 
## Dataset

The dataset is available from the AIcrowd page, as linked in the PDF project description
Download the provided datasets at : https://www.aicrowd.com/challenges/epfl-ml-text-classification/dataset_files.

## Additional Data

For reasons of data size, you can find additional the additional data that we used and we created on this Drive directory :

https://drive.google.com/drive/folders/1B1npxMavHWLGdUM_cgNZ7DCLNERD7NEy?usp=sharing

There is our trained embeddings, the already preprocessed data, and the trained BERT model used to make the submissions. 

In addition to this, the BERT model has been hosted on the `Hugging Face` platform, in order to be used more easily with the `transformers` library.
Link on the Hugging Face platform of our BERT model : https://huggingface.co/adam-chell/tweet-sentiment-analyzer


## Library requirements

In order to run our entire code, you will need the following librairies installed :

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `nltk`
- `gensim`
- `fasttext`
- `sklearn`
- `re`
- `keras` with backend `tensorflow` installed and configured
- `torch`
- `transformers`
- `sys`

## Files

Architecture of the files in the repo :

*Note : for the word embeddings notebooks, you can either run it entirely which will create the word embedding localy, or import the trained embeddings we provide on the Drive directory, and only run the other cells*

* `notebooks` : 
    *  `Bow.ipynb` : performs representation learning of the text, and train 4 ML baseline classifiers and outputs the performance on validation set
    *  `TF-IDF.ipynb` : performs representation learning of the text, and train 4 ML baseline classifiers and outputs the performance on validation set. !requires to setup the $data$ directory (see below)
    *  `Word2Vec.ipynb` : create the word embedding, and train 4 ML baseline classifiers and outputs the performance on validation set
    *  `FastText.ipynb` : create the word embedding, and train 4 ML baseline classifiers and outputs the performance on validation set
    *  `GloVe.ipynb` : imports the pre-trained embedding, and train 4 ML baseline classifiers and outputs the performance on validation set. !requires to setup the $data$ directory (see below)
    *  `lstm.ipynb` : train an LSTM network with tuned hyperparameters, and output accuracy on validation set (training will be less than 5 minutes if run on colab, else expect ~25 minutes). !requires to setup the $data$ directory (see below)
    *  `BERT.ipynb` : finetunes a pretrained BERT model on the dataset (this is a notebook runned on Colab, if you want to run it, we strongly advice you to do the same.

* `src` : 
    * `load_utils.py` : loads the data from the files into dataframes
    * `preprocessing.py` : pre-processing steps taken
    * `preprocessing.py` : compute main metrics and outputs confusion matrix for the predicted values
    * `Tweet.py` : creates a PyTorch dataset for the BERT model

- `run.py` : creates the .csv file used in our best prediction on AIcrowd. Make sure you have transformers library installed, and Tweet.py in the same directory.

- `Tweet.py` : creates a PyTorch dataset for the BERT model. It is also present in the  `src` directory.

* `data` : Contains some embedding & dataframe for our training/validation Should be initialy empty, but you will need to fill it as described bellow:
  * `cleanedDataframe`: contains the cleaned_dataframe (2 version: one lemmatized, one not). You can download the 2 files from [here](https://drive.google.com/drive/folders/1mfRpJSUcPzl-PN04SHkjci514wlkRAPq)
  * `word_Embeddings`:  contains the word embedding for fastText & glove. You can download FastText and pre_trained_glove_embedding directories [here](https://drive.google.com/drive/folders/1ZAduIqmeDOEpySYyiDZsKb2RPIUDhtsD)  
  - `data_submission_preprocessed.csv`: the data for the submission, already pre processed. This csv will be used to make all the AI-crowd submissions.


Please do not change name of directories/files, otherwise you will not be able to execute the notebooks.



