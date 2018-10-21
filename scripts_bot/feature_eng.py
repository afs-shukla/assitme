import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def get_intent_data():
    df = pd.read_csv('datasets/intent_data.csv')
    return df

def data_prepare():
    col = ['class', 'question']
    y=get_intent_data()
    y = y[col]
    y = y[pd.notnull(y['question'])]
    y.columns = ['class', 'question']
    y['category_id'] = y['class'].factorize()[0]
    category_id_df = y[['class', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'class']].values)
    return y
    
def naive_algo():
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    df=data_prepare()
    features = tfidf.fit_transform(df.question).toarray()
    labels = df.category_id
    features.shape
    X_train, X_test, y_train, y_test = train_test_split(df['question'], df['class'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    return clf,count_vect

def predict(question):
    clf,count_vect=naive_algo()
    intent=clf.predict(count_vect.transform([question]))
    intent=str(intent).strip("['']")
    return intent

def prepare_entity_dict():
   entity_dict={}
   with open("datasets/entity_data.txt","r",encoding='UTF8') as entity:
        for row in entity.readlines():
            print("prepare_entity_dict row = ",row)
            entity_dict.update({row.rstrip().lower(): row.rstrip().lower()})
   entity.close()
   return entity_dict     
        

#ques=input("Enter your question ")
#x=predict(ques)
#print(str(x).strip("['']"))

