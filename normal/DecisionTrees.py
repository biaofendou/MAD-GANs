from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import classification_report
import swat

iris = load_iris()
X, Y = swat.swat_test()
# print(X)
# print(Y)
testX, testY = swat.swat_test()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=51)
# decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree = decision_tree.fit(X[:300000], Y[:300000].astype('int'))
res = decision_tree.predict(testX[300000:])
count = 0
for i in res:
    if i == 0:
        count += 1
print(count)

print(classification_report(testY[300000:].astype('int'), res))
