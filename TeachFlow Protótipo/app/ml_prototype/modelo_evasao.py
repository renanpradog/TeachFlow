import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = {
    'frequencia': [90, 70, 80, 60, 95, 40, 30, 85],
    'nota_media': [8, 6, 7, 5, 9, 4, 3, 8],
    'atividades_entregues': [10, 8, 9, 5, 12, 4, 2, 11],
    'evadiu': [0, 0, 0, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df.drop('evadiu', axis=1)
y = df['evadiu']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)


novo_aluno = pd.DataFrame({'frequencia': [65], 'nota_media': [5], 'atividades_entregues': [6]})
prob_evasao = clf.predict_proba(novo_aluno)[0][1]
print(f"Probabilidade de evasão: {prob_evasao:.2%}")


