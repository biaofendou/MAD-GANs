from sklearn.ensemble import RandomForestClassifier
import swat
from sklearn.metrics import classification_report

X, Y = swat.swat_test()
clf = RandomForestClassifier(n_estimators=10)#随机森林中树的颗数：n_estimators，越大越好，默认为10
clf = clf.fit(X[:300000], Y[:300000].astype('int'))
res = clf.predict(X[300000:])
print(classification_report(Y[300000:].astype('int'), res))
