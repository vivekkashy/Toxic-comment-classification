# Toxic-comment-classification

Introduction:

Problem Statement
The threat of abuse and harassment can be seen more often on online or social media platform that have
numerous effect ranging from psychological to sociological and more to individuals. Many users restraining themselves to give any
feedback on online platform and that can have big impact for businesses or any other sector that relies upon the
users feedback for their services.
Basically comments are divided into three categories that is Positive, negative and neutral and negative
categories are divided into different categories depending on their negativity scores like threat, obscene, insult,
toxic ,identity hate, etc.
In this project we have given the task to prepare a multi label classification model that could be able to analyze
and classify the Wikipedia’s talk page into its respective categories

Data
Wikipedia’s talk page data on which model has to be build. It contains the comment ID, comments along with its respective label
like toxic, severe, etc.

Methodology
Basic Feature Extraction:
We can use data to extract number of features from it like the number of words, number of character, Average
word length, Number of stopwords, Number of special character, Number of numeric, Number of upper case
words. Using these analysis we get the information about the content of the document which could be helpful
for the next steps that is data preprocessing.
Let’s read the training dataset in order to perform different task on it.

Pre Processing
This process involves the preprocessing of raw data to prepare it for further analyses, basically in this process
we transform the data to a required format so that the machine learning model could be built upon it.
So for better result it’s better to convert the raw data into clean data because Machine learning model needs
data in a specified format. Another purpose of cleaning data is that more than one algorithm could built upon it.
It is used for extracting interesting and non-trivial knowledge from unstructured data .The objective of this
study is to analyze the issue of preprocessing methods such as Tokenization, stopwords removal and stemming
for the text document keywords.
Text documents are preprocessed because the text data contains number, dates, and the common words such as
prepositions, articles etc. that are unlikely to give much information. It reduces data file size of the text
documents and improve the efficiency and effectiveness of the model. One by one we will preprocess the data
according to the model requirement.

Advance preprocessing.
Term frequency (Tf):
Term frequency is simply the ratio of the count of a word present in a sentence, to the length of the sentence.
Tf = (Number of times term T appears in the particular row) / (number of terms in that row).
Inverse document frequency (idf):
The idea behind this method is that some words occur more than one document frequently and that type words
are not much important for our analysis therefore we use idf to penalize the occurrence of these words.
Idf= log (N/n),
where, N is the total number of rows and n is the number of rows in which the word was present.
The more the value of IDF, the more unique is the word.
Term Frequency – Inverse Document Frequency (TF-IDF)
TF-IDF is the multiplication of the TF and IDF which we calculated above.

tf-idf(d, t) = tf(t) * idf(d, t)

idf(d, t) = log [ n / df(d, t) ] + 1, 

The effect of adding "1" to the idf in the equation above is that terms with zero idf, i.e., terms that occur in all documents in a training set, will not be entirely ignored.
N-grams:
N-gram is the combination of word together. The purpose behind this is to know the structure of the sentence
like what word or letter are likely to follow other words or letter.
N=1, is called unigram, N=2 are bigram, N=3 are trigram and so on. The longer the N-gram, the more context it
can capture but if N-gram is too long it will fail to capture the basic information of words.

Modelling
Modelling and evaluation:
Now we have the data in proper format which we can use to feed it into machine learning algorithm. Our data is
in form of matrix where all the words are a feature and the values are tf-idf that we have calculated earlier. Now
we will build multiple machine learning model on top of this data and we will compare the accuracy score of
each algorithm t determine which model we can use for multi-label classification. Since it is multi –label
classification problem therefore we have to give input of the target variable in a sequence so that it could
predict the target variable one by one.
First we will build a model using Naïve-Bayes algorithm because Naive Bayes classifiers, a family of
classifiers that are based on the popular Bayes’ probability theorem, are known for creating simple yet well
performing models, especially in the fields of document classification and disease prediction.
Then we will build a model using logistic-regression algorithm since it performs very well when we have large
amount of text data.
Then our last model will be a hybrid model using Naïve-Bayes log-count ratio as an additional feature as an
input to logistic regression, this is known as NB-LR baseline and performs equally well like any other state of
the art model.
Naive-Bayes. It performs very well short length of text data. We will apply this algorithm and check how
it is performing on our data using different types of performance matrix.
Logistic Regression:
When we have to deal with high dimension data logistic regression outperforms tree based classification and
text mining is such that area where we have high dimensional data and it performs exceptionally well on these
data. We will now implement logistic regression algorithm to check whether it can perform well than Naïve-
Bayes or not.

NB-LR Baseline
Now proceeding to our next model that is NB-LR baseline that works Naïve-Bayes log-count ratio as an
additional feature as an input to logistic regression model, this is known as NB-LR baseline and performs
equally well like any other state of the art model.

Log count ratio (r) =Ratio of feature ‘f’ in document (with label=1)/Ratio of feature ‘f’ in document (with
label=0).

Where,

Ratio of feature ‘f’ in document (with label=1) =Number of times a document (with label=1) contains a feature
‘f’ divided by the number of document (with label=1)

Ratio of feature ‘f’ in document (with label=0) =Number of times a document (with label=0) contains a feature
‘f’ divided by the number of document (with label=0)

Mentioned Below are the basic Naïve-Bayes feature extraction function.

def pr(y_i, y):

p = x[y==y_i].sum(0)

return (p+1) / ((y==y_i).sum()+1)

Where,

Y_i = Label

y = Multi label categories in the training data

x=Training data in matrix form.

Model selection
We have built three model and tested the accuracy of all the model now we will analyse which model we
should select for the deployment.
Comparison on the basis of Accuracy:
Accuracy of all the models are above 90% hence all the models. Acceptable in deployment but LR and NB-LR
has an edge Over NB.
Now we will analyse into the classification reports of all the models:
Comparing all the classification reports of all the models we can say that:
In class toxic LR is performing well than NB-LR and NB.
In class severe_toxic LR is performing better than all
In class obscene LR outperforms all other models
In class Threat NB-LR is doing well than other two models.
In class Insult LR is marginally good over other models
In class Identity hate LR is better than other two models
Now we have analysed accuracy and classification reports of all the models therefore we can select either LR or
NB-LR because both the model are performing outstanding on the text dataset with accuracy of more than 95%.

















