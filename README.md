# SRO6.2
# Импортируем необходимые библиотеки
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузим набор данных (например, набор данных Iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создадим дерево решений
clf = DecisionTreeClassifier()

# Обучим дерево решений на обучающем наборе
clf.fit(X_train, y_train)

# Сделаем прогнозы на тестовом наборе данных
y_pred = clf.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели:", accuracy)
