import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.DESCR)

diabetes_x = diabetes.data[:,np.newaxis,2]

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_x_train,diabetes_y_train)

diabetes_y_predected = model.predict(diabetes_x_test)
print("Mean squared error = ", mean_squared_error(diabetes_y_test,diabetes_y_predected))
print("Coef",model.coef_)
print("Coef",model.intercept_)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predected)
plt.show()

