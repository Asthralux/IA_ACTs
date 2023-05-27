import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    #El __init__ es como un constructor pero este no crea el objeto... se utiliza para inicializar un objeto después de que ha sido creado
    def __init__(self):
        # Conjunto de datos hardcodeados
        self.data = np.array([[1, 651, 23], [2, 762, 26], [3, 856, 30], [4, 1063, 34], [5, 1190, 43],
                              [6, 1298, 48], [7, 1421, 52], [8, 1440, 57], [9, 1518, 58]])
        # Calcular los coeficientes Beta_0 y Beta_1 para la regresión lineal simple
        self.beta_0, self.beta_1 = self.linear_fit(self.data)

    def linear_fit(self, data):
        # Extraer las coordenadas X e Y del conjunto de datos
        X = data[:, 2]  # Advertising
        Y = data[:, 1]  # Sales

        # Calcular las sumas necesarias
        n = len(X)
        x_sum = np.sum(X)
        y_sum = np.sum(Y)
        xy_sum = np.sum(X * Y)
        x_squared_sum = np.sum(X ** 2)

        # Calcular los coeficientes Beta_0 y Beta_1
        beta_1 = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum ** 2)
        beta_0 = (y_sum - beta_1 * x_sum) / n
        #print(beta_1) Es la pendiente
        return beta_0, beta_1

    def predict(self, x):
        # Predecir el valor Y utilizando la ecuación de la regresión lineal simple
        return self.beta_0 + self.beta_1 * x

    def calculate_r_squared(self):
        # Calcular el coeficiente de determinación R^2
        Y = self.data[:, 1]
        y_mean = np.mean(Y)
        y_pred = self.predict(self.data[:, 2])

        SSE = np.sum((Y - y_pred) ** 2)
        SST = np.sum((Y - y_mean) ** 2)

        return 1 - (SSE / SST)

    def plot(self):
        # Graficar la línea de la regresión lineal simple y los puntos del conjunto de datos
        x = np.linspace(np.min(self.data[:, 2]) - 1, np.max(self.data[:, 2]) + 1, 100)
        y = self.predict(x)

        plt.scatter(self.data[:, 2], self.data[:, 1], color='red', label='Data points')
        plt.plot(x, y, color='black', label='Linear fit')
        plt.xlabel('Advertising')
        plt.ylabel('Sales')
        plt.legend()
        plt.title('Simple Linear Regression')
        plt.show()

    def __str__(self):
        # Devolver la ecuación de la regresión lineal simple como una cadena de texto
        return f"ŷ = {self.beta_0:.2f} + {self.beta_1:.2f}x"


if __name__ == "__main__":
    # Crear una instancia de la clase SimpleLinearRegression
    slr = SimpleLinearRegression()

    # Imprimir la ecuación de la regresión lineal simple
    print("Ecuación de regresión lineal simple:")
    print(slr) #Imprime Y de hat igual a B0 + la pendiente multiplicado por el valor Input

    # Predecir un valor de Y a partir de un valor X de entrada
    x_input = float(input("Ingrese un valor de Advertising para predecir Sales: "))
    y_predicted = slr.predict(x_input)
    print(f"Valor de Sales predecido: {y_predicted}")

    # Calcular e imprimir el coeficiente de determinación R^2
    r_squared = slr.calculate_r_squared()
    print(f"Coeficiente de determinación R^2: {r_squared}")

    # Graficar la línea de la regresión lineal simple y los puntos del conjunto de datos
    slr.plot()