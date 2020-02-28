import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# lee los numeros
numeros = skdata.load_digits()

# lee los labels
target = numeros['target']

# lee las imagenes
imagenes = numeros['images']

# cuenta el numero de imagenes total
n_imagenes = len(target)

# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Split en train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)

# todo lo que es diferente de 1 queda marcado como 0
y_train[y_train!=1]=0
y_test[y_test!=1]=0


# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

# numero de componentes a utilizar, de 3 a 40.
n_components = 10

# arrays para guardar valores de f1
#f1_test_unos = np.ones(len(n_comp))
#f1_train_unos = np.ones(len(n_comp))
#f1_test_otros = np.ones(len(n_comp))
#f1_train_otros = np.ones(len(n_comp))
# encuentro los parametros del clasificador


linear.fit(x_train_transform[:,:n_components], y_train)

# predigo los valores para train
y_predict_train = linear.predict(x_train_transform[:,:n_components])

# predigo los valores para test
y_predict_test = linear.predict(x_test_transform[:,:n_components])

prob_test=linear.predict_proba(x_test_transform[:,:n_components]) #es un vector de dos columnas con la probabilidad de cada
#clase: cero o uno.

precision_and_recall=sklearn.metrics.precision_recall_curve(y_test,prob_test[:,1],pos_label=1)
precision_uno=precision_and_recall[0]
recall_uno=precision_and_recall[1]
threshold_uno=precision_and_recall[2]

f1_uno=2*(precision_uno*recall_uno)/(precision_uno+recall_uno)
max_f1_uno=max(f1_uno)
max_p_uno=threshold_uno[np.where(f1_uno==max(f1_uno))]
# Encuentro los autovalores y autovectores de las imagenes marcadas como 1.
numero = 1
dd = y_train!=numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)

# pueden ser complejos por baja precision numerica, asi que los paso a reales
valores = np.real(valores)
vectores = np.real(vectores)

# reordeno de mayor a menor
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

# encuentro las imagenes en el espacio de los autovectores
x_test_transform = x_test @ vectores
x_train_transform = x_train @ vectores

# inicializo el clasificador
linear = LinearDiscriminantAnalysis()

# numero de componentes a utilizar, de 3 a 40.
n_components = 10

# arrays para guardar valores de f1
#f1_test_unos = np.ones(len(n_comp))
#f1_train_unos = np.ones(len(n_comp))
#f1_test_otros = np.ones(len(n_comp))
#f1_train_otros = np.ones(len(n_comp))
# encuentro los parametros del clasificador


linear.fit(x_train_transform[:,:n_components], y_train)

# predigo los valores para train
y_predict_train = linear.predict(x_train_transform[:,:n_components])

# predigo los valores para test
y_predict_test = linear.predict(x_test_transform[:,:n_components])

prob_test=linear.predict_proba(x_test_transform[:,:n_components]) #es un vector de dos columnas con la probabilidad de cada
#clase: cero o uno.

precision_and_recall=sklearn.metrics.precision_recall_curve(y_test,prob_test[:,1],pos_label=1)
precision_cero=precision_and_recall[0]
recall_cero=precision_and_recall[1]
threshold_cero=precision_and_recall[2]

f1_cero=2*(precision_cero*recall_cero)/(precision_cero+recall_cero)
plt.figure()

plt.subplot(1,2,1)
max_f1_cero=max(f1_cero)
max_p_cero=threshold_cero[np.where(f1_cero==max(f1_cero))]
plt.plot(threshold_cero,f1_cero[:-1],label='ceros')
plt.plot(max_p_cero,max_f1_cero,'ro',markersize=10)
plt.plot(threshold_uno,f1_uno[:-1],label='unos')
plt.plot(max_p_uno,max_f1_uno,'ro',markersize=10)
plt.xlabel('Probabilidad')
plt.ylabel('f1')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall_cero,precision_cero,label='ceros')
plt.plot(recall_uno,precision_uno,label='unos')
plt.xlabel('Cobertura')
plt.ylabel('Precisión')
plt.legend()

plt.subplots_adjust(wspace=0.5)
plt.savefig('F1_prec_recall.png')
