import numpy as np
import pandas as pd
import re
import nltk
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from itertools import combinations
from nltk.tag import DefaultTagger
py_tag = DefaultTagger ('NN')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

from sklearn import naive_bayes, metrics,svm ,tree
import matplotlib.pyplot as plt
import seaborn as sns


##############################################################################################################################################################
##############################################################################################################################################################

# import data
movies_data=pd.read_csv('movies.csv',sep=";", encoding='mac_roman')
users_data = pd.read_csv('users.csv',sep=";")
ratings_data = pd.read_csv('ratings.csv',sep=";")

movies_data['title'] = movies_data['title'].astype(str)
movies_data['genres'] = movies_data['genres'].astype(str)

##############################################################################################################################################################
##############################################################################################################################################################

# there are duplicate rows or not
DuplicateNumber=movies_data.duplicated().sum()
# print(DuplicateNumber)

DuplicateNumber=users_data.duplicated().sum()
# print(DuplicateNumber)

DuplicateNumber=ratings_data.duplicated().sum()
# print(DuplicateNumber)

##############################################################################################################################################################
##############################################################################################################################################################

# there is null or not
NullNumber=movies_data.isnull().sum()
# print(NullNumber)

#there is null or not
NullNumber=users_data.isnull().sum()
# print(NullNumber)

#there is null or not
NullNumber=ratings_data.isnull().sum()
# print(NullNumber)

##############################################################################################################################################################
##############################################################################################################################################################

movies_data["Unnamed: 3"] =movies_data['genres']
movies_data = movies_data.rename(columns = {"Unnamed: 3":"oldGenres"})

###############################################################################
###############################################################################

# in content based methodology we will use generes and movie title 
# as content tag recommend based on the similarly of this tags
content_df = movies_data[['title','genres']]

#combine the content in one coulmn
content_df['content']=content_df['title']+" "+content_df['genres']

##############################################################################################################################################################
##############################################################################################################################################################

# to lower case
movies_data['title'] = movies_data['title'].str.lower()
movies_data['genres'] = movies_data['genres'].str.lower()
content_df['content'] = content_df['content'].str.lower()

###############################################################################
###############################################################################

# function that removes numerical values using regular expressions
def remove_numbers(text):
    pattern = r'\d+'
    filtered_text = re.sub(pattern, '', text)
    return filtered_text

# remove numerical values from title
movies_data['title'] = movies_data['title'].apply(remove_numbers)
content_df['content'] = content_df['content'].apply(remove_numbers)

###############################################################################
###############################################################################

# function that removes  punctuation and special characters using regular expressions
def remove_punctuation(text):
    pattern = r'[^\w\s]'
    filtered_text = re.sub(pattern, ' ', text)
    return filtered_text

# remove punctuation and special characters from title, genres
movies_data['title'] = movies_data['title'].apply(remove_punctuation)
movies_data['genres'] = movies_data['genres'].apply(remove_punctuation)
content_df['content'] = content_df['content'].apply(remove_punctuation)

###############################################################################
###############################################################################

# function that removes stopwords using NLTK
stopwords = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# remove stopwords from title
movies_data['title'] = movies_data['title'].apply(remove_stopwords)
content_df['content'] = content_df['content'].apply(remove_stopwords)

###############################################################################
###############################################################################

# tokenizer function using NLTK
def tokenize_text(text):
    return nltk.word_tokenize(text)

# tokenize title, genres
movies_data['title'] = movies_data['title'].apply(tokenize_text)
movies_data['genres'] = movies_data['genres'].apply(tokenize_text)
content_df['content']= content_df['content'].apply(tokenize_text)

##############################################################################################################################################################
##############################################################################################################################################################

# split the col into two 
users_data['zip-code'], users_data['zip-code2']=users_data['zip-code'].str.split('-', 2).str
del users_data['zip-code2']

users_data['zip-code']= users_data['zip-code'].astype(int)

###############################################################################
###############################################################################

# split the col into two 
content_df['title'], content_df['title2']=content_df['title'].str.split(', ', 1).str
del content_df['title2']

###############################################################################
###############################################################################

# Scale the data using Normalization in zip-code,  timestamp
my_column_array = np.array(users_data['zip-code'])
reshaped_data = my_column_array.reshape(-1, 1)
users_data['zip-code'] = MinMaxScaler().fit_transform(reshaped_data)

my_column_array = np.array(ratings_data['timestamp'])
reshaped_data = my_column_array.reshape(-1, 1)
ratings_data['timestamp'] = MinMaxScaler().fit_transform(reshaped_data)

##############################################################################################################################################################
##############################################################################################################################################################

# encoding gender --> M=1, F=0
users_data = pd.get_dummies(users_data, columns=['gender'], drop_first=True)

##############################################################################################################################################################
##############################################################################################################################################################

# stemming

def stemming_text(text):
    stemmer=PorterStemmer()
    return [stemmer.stem(word) for word in text]

# movies_data['title'] = movies_data['title'].apply(stemming_text)
# movies_data['genres'] = movies_data['genres'].apply(stemming_text)
content_df['content']=content_df['content'].apply(stemming_text)

##############################################################################################################################################################
##############################################################################################################################################################

# lemmatization

tag_map={
    'JJ':wordnet.ADJ,
    'JJR':wordnet.ADJ,
    'JJS':wordnet.ADJ,
    'RP':wordnet.ADJ,

    'NN':wordnet.NOUN,
    'NNS':wordnet.NOUN,
    'NNP':wordnet.NOUN,
    'NNPS':wordnet.NOUN,    
    'CD':wordnet.NOUN,
    'CC':wordnet.NOUN, 
    'PRP':wordnet.NOUN,
    'PRP$':wordnet.NOUN,
    'MD':wordnet.NOUN,
    'DT':wordnet.NOUN,
    'FW':wordnet.NOUN,
    'WDT':wordnet.NOUN,
    'WRB':wordnet.NOUN,
    'WP':wordnet.NOUN,
    'TO':wordnet.NOUN,
    'UH':wordnet.NOUN,
    
    'VB':wordnet.VERB,
    'VBB':wordnet.VERB,
    'VBG':wordnet.VERB,
    'VBN':wordnet.VERB, 
    'VBP':wordnet.VERB,
    'VBZ':wordnet.VERB,
    
    'VBD':wordnet.ADV,
    'IN':wordnet.ADV,
    'RB':wordnet.ADV, 
    'RBR':wordnet.ADV, 
    'RBS':wordnet.ADV,
    
    
   
}



def tag_text(text):    
    return pos_tag(text)

movies_data['title'] = movies_data['title'].apply(tag_text)
movies_data['genres'] = movies_data['genres'].apply(tag_text)

def lemmatization_text(text):
    lemmatizer=WordNetLemmatizer()
    return [lemmatizer.lemmatize(word[0],pos=tag_map[word[1]]) for word in text]

movies_data['title'] = movies_data['title'].apply(lemmatization_text)

movies_data['genres'] = movies_data['genres'].apply(lemmatization_text)

##############################################################################################################################################################
##############################################################################################################################################################

#Splitting data into train and validation

np.random.seed(42)
# Randomly select 3883 samples from ratings_data
idx = np.random.choice(len(ratings_data['rating']), size=len(movies_data['oldGenres']), replace=False)
ratings_subset = ratings_data['rating'].iloc[idx]

# Use train_test_split() 
train_x, valid_x, train_y, valid_y = train_test_split(movies_data['oldGenres'], ratings_subset, test_size=0.2, random_state=42)


# create object
vectorizer = CountVectorizer()

# fit and transform using the CountVectorizer object
train_x_bow = vectorizer.fit_transform(train_x)
valid_x_bow = vectorizer.transform(valid_x)

bowVocab=vectorizer.vocabulary_
bowArray_train_x=train_x_bow.toarray()
bowArray_valid_x=valid_x_bow.toarray()

bowDataSet_train_x=pd.DataFrame(bowArray_train_x,columns=vectorizer.get_feature_names())
bowDataSet_valid_x=pd.DataFrame(bowArray_valid_x,columns=vectorizer.get_feature_names())

# Define array of column names
valid_columns = [ 'action', 'adventure','animation', 'children', 'comedy','crime', 'documentary', 'drama'	,'fantasy','film-noir','horror','musical','mystery','romance','sci-fi','thriller','war','western']

# Drop columns that are not in the valid_columns array
drop_columns_train_x = set(bowDataSet_train_x) - set(valid_columns)
drop_columns_vaild_x = set(bowDataSet_valid_x) - set(valid_columns)

bowDataSet_train_x=bowDataSet_train_x.drop(columns=drop_columns_train_x, inplace=False)
bowDataSet_valid_x=bowDataSet_valid_x.drop(columns=drop_columns_vaild_x, inplace=False)

##############################################################################################################################################################
##############################################################################################################################################################

#convert content coulmn form list to a corpus 
content_df['content']=content_df['content'].apply(lambda x:' '.join(x))

# create object
vectorizer = CountVectorizer()
vector=vectorizer.fit_transform(content_df['content']).toarray()

cv_tfidf=TfidfVectorizer(stop_words='english')
tfidf = cv_tfidf.fit_transform(content_df['content'])

similarity =cosine_similarity(vector)

#cosine similarity with tfidf
tfdf_similarity =cosine_similarity(tfidf)

# make recommender more friendly and add movie title only 
# since if we remove all numbers this will lead to dublicates if a movie has more than one chapter 
def remove_date(title):
    pattern=r'\([^()]*\)' 
    pure_title=re.sub(pattern,'',title)
    return pure_title

content_df['title']=content_df['title'].apply(remove_date)


def Recommender_using_countvictorizer(movie):
    movie_indx=content_df[content_df['title']==movie].index[0]
    distance=similarity[movie_indx]
    top_movies=sorted(list(enumerate (distance)) , reverse=True, key=lambda x:x[1])[1:11] #top 10
    for i in top_movies:
        #since top_movies is list of tuple contain movie index, and it's similarity value
        print(content_df['title'][i[0]])
        #print ("indx of movie",i[0])
        
        
def Recommender_using_Tfidf(movie):
    movie_indx=content_df[content_df['title']==movie].index[0]
    distance=tfdf_similarity[movie_indx]
    top_movies=sorted(list(enumerate (distance)) , reverse=True, key=lambda x:x[1])[1:11] #top 10
    for i in top_movies:
        #since top_movies is list of tuple contain movie index, and it's similarity value
        print(content_df['title'][i[0]])


print("........................................................................")
Recommender_using_countvictorizer('Jumanji ')
print("........................................................................")
Recommender_using_Tfidf('Jumanji ')
print("........................................................................")

##############################################################################################################################################################
##############################################################################################################################################################


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)

# Naive Bayes trainig
NBaccuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2), train_x_bow , train_y,valid_x_bow )
print ("Accuracy of Naive Bayes :  ", NBaccuracy)

# SVM model trainig
SVMaccuracy = train_model(svm.SVC(kernel='rbf', random_state=0 , gamma=1 ,C=0.1), train_x_bow, train_y,valid_x_bow )
print ("Accuracy of SVM :  ", SVMaccuracy)

#Decision tree
DTaccuracy = train_model(tree.DecisionTreeClassifier(), train_x_bow, train_y,valid_x_bow )
print ("Accuracy of Decision Tree :  ", DTaccuracy)


##############################################################################################################################################################
##############################################################################################################################################################


# # Naive Bayes
# nb_model = naive_bayes.MultinomialNB(alpha=0.2)
# nb_model.fit(train_x_bow, train_y)
# nb_pred = nb_model.predict(valid_x_bow)

# # SVM
# svm_model = svm.SVC(kernel='rbf', random_state=0, gamma=1, C=0.1)
# svm_model.fit(train_x_bow, train_y)
# svm_pred = svm_model.predict(valid_x_bow)

# # Decision Tree
# dt_model = tree.DecisionTreeClassifier()
# dt_model.fit(train_x_bow, train_y)
# dt_pred = dt_model.predict(valid_x_bow)


# # Classification Report
# print("Naive Bayes Classification Report:\n", metrics.classification_report(valid_y, nb_pred))
# print("SVM Classification Report:\n", metrics.classification_report(valid_y, svm_pred))
# print("Decision Tree Classification Report:\n", metrics.classification_report(valid_y, dt_pred))


##############################################################################################################################################################
##############################################################################################################################################################


# Create bar chart
labels = ['Naive Bayes', 'SVM', 'Decision Tree']
accuracy = [NBaccuracy, SVMaccuracy, DTaccuracy]

x_pos = np.arange(len(labels))
plt.bar(x_pos, accuracy, align='center', alpha=0.5)
plt.xticks(x_pos, labels)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()







