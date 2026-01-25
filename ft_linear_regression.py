import numpy as np

def setData():
    data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
    mileage = data[:, 0]
    price = data[:, 1]

    # Normalize mileage
    mileage = (mileage - np.mean(mileage)) / np.std(mileage)

    Theta = np.zeros(2)
    return mileage, price, Theta


def compute_cost(Theta, mileage, price):
    m = len(mileage)
    predictions = Theta[0] + Theta[1] * mileage
    error = predictions - price
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    return cost


def estimatePrice(Theta, mileage) :
    return Theta[0] + (Theta[1] * mileage)

def train(Theta, mileage, price, learning_rate) :
    m = len(mileage)
    for j in range (1000) :
        sum_theta1 = 0
        sum_theta0 = 0
        for i in range(len(mileage)):
            calculated_price = estimatePrice(Theta, mileage[i])
            error = calculated_price - price[i]
            sum_theta0 += error
            sum_theta1 += error * mileage[i]
        tmp_theta0 = Theta[0] - learning_rate * (1 / m) * sum_theta0
        tmp_theta1 = Theta[1] - learning_rate * (1 / m) * sum_theta1

        Theta[0] = tmp_theta0
        Theta[1] = tmp_theta1
        cost = compute_cost(Theta, mileage, price)
        print("Final cost:", cost)
        print(f"\nstep: {j}   theta0 = {Theta[0]}    theta1 = {Theta[1]}")
    return Theta

def ft_linear_regression() :
    learning_rate = 0.001
    mileage, price, Theta = setData()
    Theta = train(Theta, mileage, price, learning_rate)
    return Theta

def main() :
    Theta = ft_linear_regression()

if __name__ == "__main__" :
    main()