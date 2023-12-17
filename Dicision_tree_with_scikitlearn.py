import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# Create a DataFrame using the feature matrix (data) and feature names
dataset = pd.DataFrame(data.data, columns=data.feature_names)
#dataset.to_csv('breast_cancer_dataset.csv', index=False)
#print("CSV file created successfully.")
from sklearn.model_selection import train_test_split
x= dataset.copy()#we are using the whole dataset features for the prediction
y=data['target']#our tagrget is to find wheather the person is malignant or fine with breast cancetr, so the target will be 1 or 0
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier() #defalut way
clf=DecisionTreeClassifier(max_depth=4) #how depth the tree can be
#clf= DecisionTreeClassifier(ccp_alpha=0.01)# this is for pruning which will increase the accuracy by reducing overfitting
#print(clf.get_params())  #we will get all the parameters used for decision tree classifier, will show default values initially.
clf =clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
#print("CHANCES:::::",predictions)
probable_prediction =clf.predict_proba(x_test) # this will give the probablity of each of the classes
#print("PROBA:::", probable_prediction)
#PERFORMANCE_MATRICES
from sklearn.metrics import accuracy_score
print("accuracy score is =", accuracy_score(y_test,predictions))
from sklearn.metrics import confusion_matrix
print("CONFUSION MATRICS OF 0 AND 1:", confusion_matrix(y_test,predictions, labels=[0,1]))
from sklearn.metrics import precision_score
print("precision score=", precision_score(y_test, predictions))
from sklearn.metrics import recall_score
print("recall score =", recall_score(y_test,predictions))
from sklearn.metrics import classification_report
print("Classification report:.\n", classification_report(y_test,predictions, target_names=['malignant', 'benign']))
feature_names = x.columns
print(feature_names)
feature_importace = pd.DataFrame(clf.feature_importances_, index= feature_names).sort_values(0, ascending=False)
#print(feature_importace)
sns.barplot(data=feature_importace, x=feature_importace.index, y=0)
# Add title
plt.xticks(rotation=45, ha='right', fontsize=4)
#plt.figure(figsize=(8, 6))
plt.title('feature_importace')
plt.show()
from sklearn import tree
fig =plt.figure(figsize =(25,20))
_ = tree.plot_tree(clf,
                   feature_names=feature_names,
                   class_names={0:'Malignant', 1:'Benign'},
                   filled=True,
                   fontsize=10)
plt.show()
