
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()

# print(iris.DESCR)

features = iris.data
labels = iris.target
# print(features[0],labels[0])

clf = KNeighborsClassifier()
clf.fit(features,labels)

test = [[1,1,1,1]]
preds = clf.predict(test)

print(test, end='')
print(preds)