import numpy as np
import matplotlib.pyplot as plt

def generate_data(m=1000, n=100, density=0.4):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    x_star = np.random.random((m,1))
    idxs = np.random.choice(range(m), int((1-density)*m), replace=False)
    for idx in idxs:
        x_star[idx] = 0
    A = np.random.random((n, m))
    b = A.dot(x_star)
    return x_star, A, b


m, n, density = 1000, 100, 0.2
mu = 1
x_star, A, b = generate_data(m, n, density)
my_x = np.random.random((m, 1))


def f(A, x1, x2, b, mu):
    """
    对于临近梯度下降法，x1=x2，对于ADMM，将目标问题拆分为两部分
    """
    return 0.5 * np.linalg.norm(np.dot(A, x1) - b, ord=2) + mu * np.linalg.norm(x2, ord=1)


def get_gradient(A, x, b):
    return np.dot(A.transpose(), np.dot(A, x) - b)


def get_prox(learning_rate, x, mu):
    maxPart = np.abs(x) - learning_rate * mu
    
    for i in range(len(maxPart)):
        maxPart[i] = 0 if maxPart[i] <= 0 else maxPart[i]
    
    return np.sign(x) * maxPart


def step_proximal(A, x, b, mu, learning_rate):
    return get_prox(learning_rate, x - learning_rate * get_gradient(A, x, b), mu)


def step_admm(A, x1, x2, b, mu, beta, lamb, rho):
    size = np.shape(A)[1]
    newX1 = np.dot(np.linalg.inv(np.dot(A.transpose(), A) + beta * np.identity(size)), np.dot(A.transpose(), b) + beta * x2 - lamb)
    newX2 = get_prox(mu/beta, newX1+1/beta*lamb, mu)
    newLamb = lamb + rho * beta * (newX1 - newX2)

    return newX1, newX2, newLamb

def proximal_gradient_descent(A, x, b, mu, learning_rate):
    delta = 1e10
    epoch = 0
    iterationList = []
    deltaList = []

    while delta > 1e-04 and epoch < 300:
        newX = step_proximal(A, x, b, mu, learning_rate)
        delta = np.abs(f(A, newX, newX, b, mu))
        iterationList.append(epoch)
        deltaList.append(delta)
        x = newX
        epoch += 1
    
    curve, = plt.plot(iterationList, deltaList, '-')
    return curve


def admm(A, x, b, mu, beta, lamb, rho):
    x1 = x
    x2 = x
    lamb = lamb * np.ones((np.shape(A)[1], 1))
    delta = 1e10
    epoch = 0
    iterationList = []
    deltaList = []

    while delta > 1e-04 and epoch < 50:
        newX1, newX2, newLamb = step_admm(A, x1, x2, b, mu, beta, lamb, rho)
        delta = np.abs(f(A, newX1, newX2, b, mu))
        iterationList.append(epoch)
        deltaList.append(delta)
        x1 = newX1
        x2 = newX2
        lamb = newLamb
        epoch += 1

    curve, = plt.plot(iterationList, deltaList, '-')
    return curve



if __name__ == '__main__':
    curves = []
    labels = []
    for alpha in [1e-07, 3e-07, 6e-07, 1e-06, 3e-06, 6e-06]:
        curves.append(proximal_gradient_descent(A, my_x, b, mu, learning_rate=alpha))
        labels.append('alpha={0}'.format(str(alpha)))
    plt.legend(handles=curves, labels=labels)
    plt.xlabel('epoch')
    plt.ylabel('f(x)')
    plt.savefig('codes/prox.png', dpi=400)
    
    # factor = [0.1, 1]
    # curves = []
    # labels = []
    # for beta in factor:
    #     for lamb in factor:
    #         for rho in factor:
    #             curves.append(admm(A, my_x, b, mu, beta=beta, lamb=lamb, rho=rho))
    #             labels.append('beta={0}, lambda={1}, rho={2}'.format(beta, lamb, rho))
    
    # plt.legend(handles=curves, labels=labels)
    # plt.xlabel('epoch')
    # plt.ylabel('f(x)')
    # plt.savefig('codes/admm.png', dpi=400)
