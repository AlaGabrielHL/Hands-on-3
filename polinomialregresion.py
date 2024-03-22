import numpy as np

class Regresion:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(self.x)
        self.matriz_lineal = np.vstack([np.ones(self.n), self.x]).T
        self.matriz_cuadratica = np.vstack([np.ones(self.n), self.x, np.power(self.x, 2)]).T
        self.matriz_cubica = np.vstack([np.ones(self.n), self.x, np.power(self.x, 2), np.power(self.x, 3)]).T

    def calcular_coeficientes(self, matriz):
        return np.linalg.inv(matriz.T @ matriz) @ matriz.T @ self.y

    def calcular_predicciones(self, matriz, coeficientes):
        return matriz @ coeficientes

    def calcular_coef_correlacion(self, predicciones):
        return np.corrcoef(predicciones, self.y)[0, 1]

    def calcular_coef_determinacion(self, coef_correlacion):
        return coef_correlacion ** 2

    def regresion_lineal(self):
        coeficientes = self.calcular_coeficientes(self.matriz_lineal)
        predicciones = self.calcular_predicciones(self.matriz_lineal, coeficientes)
        coef_correlacion = self.calcular_coef_correlacion(predicciones)
        coef_determinacion = self.calcular_coef_determinacion(coef_correlacion)
        return coeficientes, coef_correlacion, coef_determinacion

    def regresion_cuadratica(self):
        coeficientes = self.calcular_coeficientes(self.matriz_cuadratica)
        predicciones = self.calcular_predicciones(self.matriz_cuadratica, coeficientes)
        coef_correlacion = self.calcular_coef_correlacion(predicciones)
        coef_determinacion = self.calcular_coef_determinacion(coef_correlacion)
        return coeficientes, coef_correlacion, coef_determinacion

    def regresion_cubica(self):
        coeficientes = self.calcular_coeficientes(self.matriz_cubica)
        predicciones = self.calcular_predicciones(self.matriz_cubica, coeficientes)
        coef_correlacion = self.calcular_coef_correlacion(predicciones)
        coef_determinacion = self.calcular_coef_determinacion(coef_correlacion)
        return coeficientes, coef_correlacion, coef_determinacion

    def predecir(self, valor_x):
        matriz_prediccion_lineal = np.array([1, valor_x])
        matriz_prediccion_cuadratica = np.array([1, valor_x, valor_x**2])
        matriz_prediccion_cubica = np.array([1, valor_x, valor_x**2, valor_x**3])

        coeficientes_lineal, _, _ = self.regresion_lineal()
        coeficientes_cuadratico, _, _ = self.regresion_cuadratica()
        coeficientes_cubico, _, _ = self.regresion_cubica()

        prediccion_lineal = matriz_prediccion_lineal @ coeficientes_lineal
        prediccion_cuadratica = matriz_prediccion_cuadratica @ coeficientes_cuadratico
        prediccion_cubica = matriz_prediccion_cubica @ coeficientes_cubico

        return prediccion_lineal, prediccion_cuadratica, prediccion_cubica

# Dataset
batch_size = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89]
machine_efficiency = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77 ,96 ,87 ,89 ,60 ,63 ,95 ,61 ,55, 56, 94, 93]

# Crear objeto de regresión
regresion_obj = Regresion(batch_size, machine_efficiency)

# Realizar regresiones
coeficientes_lineal, correlacion_lineal, determinacion_lineal = regresion_obj.regresion_lineal()
coeficientes_cuadratico, correlacion_cuadratico, determinacion_cuadratico = regresion_obj.regresion_cuadratica()
coeficientes_cubico, correlacion_cubico, determinacion_cubico = regresion_obj.regresion_cubica()

# Imprimir resultados
print("Regresion Lineal: y = {:.4f} + {:.4f} * Batch size".format(coeficientes_lineal[0], coeficientes_lineal[1]))
print("Coeficiente de Correlacion:", correlacion_lineal)
print("Coeficiente de Determinacion:", determinacion_lineal)

print("\nRegresion Cuadratica: y = {:.4f} + {:.4f} * Batch size + {:.4f} * Batch size^2".format(coeficientes_cuadratico[0], coeficientes_cuadratico[1], coeficientes_cuadratico[2]))
print("Coeficiente de Correlacion:", correlacion_cuadratico)
print("Coeficiente de Determinacion:", determinacion_cuadratico)

print("\nRegresion Cubica: y = {:.4f} + {:.4f} * Batch size + {:.4f} * Batch size^2 + {:.4f} * Batch size^3".format(coeficientes_cubico[0], coeficientes_cubico[1], coeficientes_cubico[2], coeficientes_cubico[3]))
print("Coeficiente de Correlacion:", correlacion_cubico)
print("Coeficiente de Determinacion:", determinacion_cubico)

# Realizar predicciones
valores_prediccion = [65, 75, 85]
for valor in valores_prediccion:
    prediccion_lineal, prediccion_cuadratica, prediccion_cubica = regresion_obj.predecir(valor)
    print(f"\nPrediccion para Batch size = {valor}:")
    print("Regresion Lineal:", prediccion_lineal)
    print("Regresion Cuadratica:", prediccion_cuadratica)
    print("Regresion Cubica:", prediccion_cubica)
