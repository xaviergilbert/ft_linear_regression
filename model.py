import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# import time
# import copy
import sys

class model_fit:
    def __init__(self, X, Y, cycle = 1000, learning_rate = 0.1):
        self.theta = [0, 0]
        self.cycle = cycle
        self.learning_rate = learning_rate / len(X)
        self.min_x = np.min(X)
        self.max_x = np.max(X)
        self.min_y = np.min(Y)
        self.max_y = np.max(Y)
        self.X = X / (self.max_x - self.min_x)
        print(self.X)
        exit()
        # self.X = (X - X.mean()) / np.mean((X - X.mean()) ** 2)
        self.Y = Y

    def mse_(self, y, y_hat):
        tmp = 0
        for arg, arg1 in zip(y_hat, y):
            tmp += (arg - arg1) ** 2
        tmp = tmp / len(y)
        return tmp

    def predict(self, X):
        predict = self.theta[1] * X + self.theta[0]
        return predict

    def ajust_learning_rate(self, mse_history, key, Y_pred):
        if key == 1:
            self.theta[0] += self.learning_rate * np.mean(Y_pred - self.Y)
            self.theta[1] += self.learning_rate * np.mean((Y_pred - self.Y) * self.X)
            self.learning_rate *= 0.1
            del mse_history[-1]
        elif key == 0:
            self.learning_rate *= 1.50

    def fit(self):
        print("Modeling...")
        mse_history = []
        for cycle in range(self.cycle):
            Y_pred = self.predict(self.X)
            self.theta[0] -= self.learning_rate * np.mean(Y_pred - self.Y)
            self.theta[1] -= self.learning_rate * np.mean((Y_pred - self.Y) * self.X)
            mse_history.append(self.mse_(self.Y, self.predict(self.X)))
            if cycle > 0:
                if mse_history[-1] >= mse_history[-2]:
                    self.ajust_learning_rate(mse_history, 1, Y_pred)
                else:
                    self.ajust_learning_rate(mse_history, 0, Y_pred)
            if self.learning_rate == 0:
                break
        self.theta[1] = self.theta[1] / (self.max_x - self.min_x)
        print("Done.")        

def main():
    data = pd.read_csv('data.csv')
    X = np.array(data[['km']])
    Y = np.array(data[['price']])
    model = model_fit(X, Y, cycle = 10000)
    model.fit()
    f = open("theta.txt", "w")
    lines_of_text = [str(model.theta[0]), "\n", str(model.theta[1]),  "\n"]
    f.writelines(lines_of_text)
    f.close()
    if len(sys.argv) > 1 and sys.argv[1] == "-graph":
        print("ici")
        predict = model.predict(X)
        plt.plot(X, Y, 'bo', X, predict, 'go')
        plt.show()

def sklearn():
    data = pd.read_csv('data.csv')
    X = np.array(data[['km']])
    Y = np.array(data[['price']])
    reg = LinearRegression().fit(X, Y)
    b = reg.intercept_
    a = reg.coef_
    print("coef :", a)
    print("score", reg.score(X, Y))
    print("intercept : ", b)
    user_input = float(input("What is the mileage of your car ?\n"))
    print("Your car is probably worth", int(user_input * a + b), "euros")


if __name__ == "__main__":
    main()
    # sklearn()