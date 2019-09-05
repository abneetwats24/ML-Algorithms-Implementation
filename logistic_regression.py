import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()

x = iris.data[:,3:]
y = (iris.target==2).astype(np.int)
# flower is iris-verginica or not
print(x, end='')
print(y)


# train a logistic regregstion classifier

clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[1.6]]))

print(example)


# using matplotlib to plot visualization

x_new = np.linspace(0,3,1000).reshape(-1,1)
print(x_new)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1],"g-",label = "verginica")
plt.show()
