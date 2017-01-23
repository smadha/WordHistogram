import load_data_set as ld

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

X_tr, X_te = [],[]

for train in ld.X_train :
    X_tr.append(" ".join([str(x) for x in train]))
    
for test in ld.X_test:
    X_te.append(" ".join([str(x) for x in test]))

print X_tr[0]
print ld.X_train[0]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_tr)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB()

model.fit(X_train_tfidf, ld.y_train)

X_new_counts = count_vect.transform(X_te)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

y_pred = model.predict(X_new_tfidf)

print classification_report(ld.y_test, y_pred)


#              precision    recall  f1-score   support
# 
#           0       0.79      0.89      0.84     12500
#           1       0.88      0.77      0.82     12500
# 
# avg / total       0.84      0.83      0.83     25000

