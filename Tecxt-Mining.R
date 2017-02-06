##### Sample code for text mining and sentiment analysis using R #####
##### Sarvesh #######

## Building a predictive model for movie review sentiment##
# Data source: http://www.cs.cornell.edu/people/pabo/movie-review-data/

install.packages(c('tm', 'SnowballC', 'wordcloud'))
library(tm)
library(SnowballC)
library(wordcloud)
library(dplyr)

reviews = read.csv("data/movie_reviews.csv", stringsAsFactors = F, row.names = 1)
# A collection of text documents is called a Corpus
review_corpus = Corpus(VectorSource(reviews$content))
# View content of the first review
review_corpus[[1]]$content

# Change to lower case, not necessary here
review_corpus = tm_map(review_corpus, content_transformer(tolower))
# Remove numbers
review_corpus = tm_map(review_corpus, removeNumbers)
# Remove punctuation marks and stopwords
review_corpus = tm_map(review_corpus, removePunctuation)
review_corpus = tm_map(review_corpus, removeWords, c("duh", "whatever", stopwords("english")))
# Remove extra whitespaces
review_corpus =  tm_map(review_corpus, stripWhitespace)

review_corpus[[1]]$content

# Sometimes stemming is necessary
review_corpus_stemmed =  tm_map(review_corpus, stemDocument)
review_corpus_stemmed[[1]]$content


# Document-Term Matrix: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Notice the dimension of the matrix
review_dtm <- DocumentTermMatrix(review_corpus)
review_dtm
inspect(review_dtm[1:10,1:10])

# Simple word cloud
findFreqTerms(review_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(review_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

# Remove the less frequent terms such that the sparsity is less than 0.95
review_dtm = removeSparseTerms(review_dtm, 0.95)
review_dtm
# The first 10 documents
inspect(review_dtm[1:10,1:20])

# tf-idf(term frequency-inverse document frequency) instead of the frequencies of the term as entries, tf-idf measures the relative importance of a word to a document
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf
# The first document
inspect(review_dtm_tfidf[1:10,1:20])

# A new word cloud
freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

# Precompiled list of words with positive and negative meanings
# Source: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
neg_words = read.table("data/negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("data/positive-words.txt", header = F, stringsAsFactors = F)[, 1]

# neg, pos contain the number of positive and negative words in each document
reviews$neg = sapply(review_corpus, tm_term_score, neg_words)
reviews$pos = sapply(review_corpus, tm_term_score, pos_words)
# remove the actual texual content for statistical models
reviews$content = NULL
# construct the dataset for models
reviews = cbind(reviews, as.matrix(review_dtm))
reviews$polarity = as.factor(reviews$polarity)

# Split to testing and training set
id_train <- sample(nrow(reviews),nrow(reviews)*0.80)
reviews.train = reviews[id_train,]
reviews.test = reviews[-id_train,]

# Compare 3 classification models
# classification tree, logistic regression, support vector machine (SVM)
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("e1071")
library(e1071) # for Support Vector Machine

reviews.tree = rpart(polarity~.,  method = "class", data = reviews.train);  
prp(reviews.tree)
reviews.glm = glm(polarity~ ., family = "binomial", data =reviews.train, maxit = 100);  
reviews.svm = svm(polarity~., data = reviews.train);

# Performance in the test set
pred.tree = predict(reviews.tree, reviews.test,  type="class")
table(reviews.test$polarity,pred.tree,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.tree, 1, 0))

pred.glm = as.numeric(predict(reviews.glm, reviews.test, type="response") > 0.5)
table(reviews.test$polarity,pred.glm,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.glm, 1, 0))

pred.svm = predict(reviews.svm, reviews.test)
table(reviews.test$polarity,pred.svm,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.svm, 1, 0))
