# Text-Classifier
Text Classifier for soft and technical skills (TalentBait Coding Test)

## Introduction
 The task was to train a model to be able to classify texts into technical skills, soft skills and texts that did not describe a skill.
 We will look at the following:
 - **Dataset**
 - **Feature Extraction**
 - **Model Training and Testing**
 - **Results**

## Dataset
The data set given was a json file containing strings with a text(in German) and a label that describes whether the text talks about soft skills, technical skills or it does not describe a skill.
I extracted the data from the file and split it into text and labels.
Also, in preparation for the training of the model, I split the dataset into training and test data (in a 3:1 ratio).

## Feature Extraction
I went with a vectorization approach to feature extraction from the text and used the TfidfVectorizor() function from the sci_kit learn library. I chose this function because it takes care of cleaning up the text and tokenization of each text before the vectorization.

## Model Training and Testing

### Training
I chose the Multinomial Naïve Bayes Classifier because it does not require great computational resources and an exceptionally large dataset to produce good results. 
After testing, I saved the model in the MNB_classifier file so it can be reused.

### Testing
I tested using the testing dataset created before. Since the train and test data was separated at the beginning, it is safe to say that the results of testing are a credible reflection of the performance of the model.

## Results
For metric evaluation I used accuracy scores, f-1 scores and confusion matrices to evaluate the performance of the model. The results can be seen in the image below:

![Image description](results.jpg)

### Analysis of Results

