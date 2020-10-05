import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class model_fit:
    def __init__(self, X, Y, cycle = 1000000, learning_rate = 0.1):
        self.theta = [0, 0]
        self.cycle = cycle
        self.learning_rate = learning_rate / len(X)
        self.min_x = np.min(X)
        self.max_x = np.max(X)
        self.min_y = np.min(Y)
        self.max_y = np.max(Y)
        self.X = X / self.max_x
        self.Y = Y / self.max_y

    def mse_(self, y, y_hat):
        tmp = 0
        for arg, arg1 in zip(y_hat, y):
            tmp += (arg - arg1) ** 2
        tmp = tmp / len(y)
        return tmp

    def predict(self, X):
        predict = self.theta[1] * X + self.theta[0]
        return predict

    def ajust_learning_rate(self, Y_pred):
        if self.mse_history[-1] >= self.mse_history[-2]:
            self.theta[0] += self.learning_rate * np.mean(Y_pred - self.Y)
            self.theta[1] += self.learning_rate * np.mean((Y_pred - self.Y) * self.X)
            self.learning_rate *= 0.1
            del self.mse_history[-1]
        else:
            self.learning_rate *= 1.50

    def fit(self):
        print("Modeling...")
        self.mse_history = []
        for cycle in range(self.cycle):
            Y_pred = self.predict(self.X)
            self.theta[0] -= self.learning_rate * np.mean(Y_pred - self.Y)
            self.theta[1] -= self.learning_rate * np.mean((Y_pred - self.Y) * self.X)
            self.mse_history.append(self.mse_(self.Y, self.predict(self.X)))
            self.ajust_learning_rate(Y_pred) if cycle > 0 else None
            if self.learning_rate == 0:
                break
        self.theta[1] = self.theta[1] * self.max_y / self.max_x
        self.theta[0] = self.theta[0] * self.max_y
        print("Done.")        

def main():
    data = pd.read_csv('data.csv')
    X = np.array(data[['km']])
    Y = np.array(data[['price']])
    model = model_fit(X, Y, cycle = 1000)
    model.fit()
    f = open("theta.txt", "w")
    lines_of_text = [str(model.theta[0]), "\n", str(model.theta[1]),  "\n"]
    f.writelines(lines_of_text)
    f.close()
    if len(sys.argv) > 1 and sys.argv[1] == "-graph":
        predict = model.predict(X)
        plt.plot(X, Y, 'bo', X, predict, 'g')
        plt.xlabel('km')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig("figure.png")

if __name__ == "__main__":
    main()