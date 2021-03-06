from __future__ import print_function

import numpy as np
from utils import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

loader = DensityLoader()
print('data loaded')

X_train, y_train = loader.load_train(flat=True)
X_test, y_test = loader.load_test(flat=True)

print('training KNN')
knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
knn.fit(X_train, y_train) 

print('testing KNN')

# distances, indices = knn.kneighbors(X_test)
y_pred = knn.predict(X_test)

real_labels = np.argmax(y_test, 1)
pred_labels = np.argmax(y_pred, 1)

acc = accuracy_score(real_labels, pred_labels)
print(acc)
print('Accuracy: {}'.format(acc))