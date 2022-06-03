#
import pandas as pd
#

# Lea el archivo `concrete.csv` y asignelo al DataFrame `df`
df = pd.read_csv("concrete.csv")  

# Asigne la columna `strength` a la variable `y`.
y = df["strength"]  

# Asigne una copia del dataframe `df` a la variable `X`.
X = df.copy()  

# Remueva la columna `strength` del DataFrame `X`.
X.drop(["strength"],axis=1, inplace=True)

print(X.shape, y.shape)

#

# Importe train_test_split
from sklearn.model_selection import train_test_split

#

# Divida los datos de entrenamiento y prueba. La semilla del generador de números
# aleatorios es 12453. Use el 75% de los patrones para entrenamiento.
(  
    x_train,  
    x_test,  
    y_train,  
    y_test,  
) = train_test_split(  
    X,  
    y,  
    test_size = 1/4,  
    random_state = 12453,  
)  

# Retorne `X_train`, `X_test`, `y_train` y `y_test`
print(x_train.shape, x_test.shape, y_train.sum().round(2), y_test.sum().round(2))


#


# Importe MLPRegressor
# Importe MinMaxScaler
# Importe Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Cree un pipeline que contenga un estimador MinMaxScaler y un estimador
# MLPRegressor
pipeline = Pipeline(
    steps=[
        (
            "minmaxscaler",
            MinMaxScaler(),
        ),
        (
            "mlpregressor",
            MLPRegressor(),
        ),
    ],
)

# Retorne el pipeline
print(pipeline.steps[0][0],
      pipeline.steps[0][1].__class__.__name__,
      pipeline.steps[1][0],
      pipeline.steps[1][1].__class__.__name__)


#

# Importe GridSearchCV
from sklearn.model_selection import GridSearchCV

# Cree una malla de búsqueda para el objecto GridSearchCV
# con los siguientes parámetros de búesqueda:
#   * De 1 a 8 neuronas en la capa oculta
#   * Activación con la función `relu`.
#   * Tasa de aprendizaje adaptativa
#   * Momentun con valores de 0.7, 0.8 y 0.9
#   * Tasa de aprendijzaje inicial de 0.01, 0.05, 0.1
#   * Un máximo de 5000 iteraciones
#   * Use parada temprana

param_grid = {
    "hidden_layer_sizes": (1,2,3,4,5,6,7,8),  
    "activation": ["relu"],  
    "learning_rate": ["adaptive"],
    "momentum": [0.7,0.8,0.9],  
    "learning_rate_init": [0.01,0.05,0.1],  
    "max_iter": [5000],  
    "early_stopping": [True],  
}

estimator = pipeline

# Especifique un objeto GridSearchCV con el pipeline y la malla de búsqueda,
# y los siguientes parámetros adicionales:
#  * Validación cruzada con 5 particiones
#  * Compare modelos usando r^2
gridsearchcv = GridSearchCV(
    estimator = estimator,
    param_grid = param_grid,
    cv = 5,
    scoring = "accuracy" 
)


print(gridsearchcv.__class__.__name__,
      gridsearchcv.cv,
      gridsearchcv.scoring,
      gridsearchcv.return_train_score)


#


# Importe mean_squared_error
from sklearn.metrics import mean_squared_error

# Obtenga el objeto GridSearchCV
estimator = gridsearchcv

# Entrene el estimador
estimator.fit(x_train, y_train)  #

# Pronostique para las muestras de entrenamiento y validacion
y_train_pred = estimator.predict(x_train)
y_test_pred = estimator.predict(x_test)  

# Calcule el error cuadrático medio de las muestras
mse_train = mean_squared_error(
    y_train,
    y_train_pred,
)

mse_test = mean_squared_error(
    y_test,
    y_test_pred,
)

# Retorne el mse de entrenamiento y prueba
print(mse_train, mse_test)
