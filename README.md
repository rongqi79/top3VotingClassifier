# top3VotingClassifier
a new ensemble classifier combined the top-3 accuracy algorithm prediction results
Supervised learning is the machine learning task of inferring a function from labeled training data. There are several supervised classification machine learning algorithms. The choice of which specific learning algorithm we should use is a critical step. The classifier’s evaluation is most often based on prediction accuracy. In this study, a new ensemble supervised classification machine learning classifier was implemented to get high prediction accuracy. This algorithm combined 5 common supervised classification machine learning algorithms (Decision tree, logistic regression, naive bayes, SVM, random forest) prediction result. For each algorithm,grid search, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm, was used for hyperparameter optimization. The best parameter set were generated after training and was used for testing. The top-3 accuracy algorithms were selected and I used voting to combine these 3 model predictions into ensemble predictions. This new class I implemented showed better accuracy in most testing cases when comparing with other algorithms and with the voting class implemented in scikit-learn.

Required Python 3 packages
Pandas,
Numpy,
Sklearn,
Scipy,
Pickle

Files included in this project:
1.VotingTop3.py
Implement algorithm which include fit, predict and score
2.Util.py
Data preprocessing
3.DataFrameImputer.py
Data preprocessing

Usage Example
1.Load data;
df = pd.read_csv('./bank/bank.csv', delimiter=';')
data_name='bank'
2.Data preprocessing;
Impute missing values.
. Columns of dtype object are imputed with the most frequent value
in column.
. Columns of other types are imputed with mean of column;
Use LabelEncoder to transform non-numerical labels to numerical labels.
optional: data normalization
df=util.dataPreprocessing(df)
3.Feature selection;
df=util.featureSelection(df)
Chisquare to drop features whose p value > 0.005;
PCA to drop features;
4.get training and test data from df
X = util.getX(df)
Y = util.getY(df)
data_train, data_test, target_train, target_test = train_test_split(X, Y,test_size=0.2)
5.Fit
clf=VotingTop3(data_name=data_name)
clf.fit(data_train,target_train)
clf.sortedAlgo(data_test,target_test)
6.Predict:
prediction=clf.predict(data_test,target_test)
7.Score
clf.score(data_test,target_test))

Result Example:
                        score  
logistic regression  0.896133  
random forrest       0.887293  
svm                  0.885083  
naive bayes          0.883978  
decision tree        0.882873  
the top accuracy of machine learning algorithms is :  logistic regression 0.896132596685
This new ensembel algorithm accuracy is  0.924861878453
