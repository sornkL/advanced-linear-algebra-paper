import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)

len_x = 1000
A = np.random.rand(len_x, len_x)
x_star = np.random.rand(len_x, 1)
b = A*x_star
my_x1 = np.random.rand(len_x, 1)
my_x2 = np.random.rand(len_x, 1)
my_x3 = np.random.rand(len_x, 1)

def get_grad(x):
    return 2*A.T*(A*x-b)

def get_loss(x):
    return np.linalg.norm(A*x-b, ord=1)**2

delta = 1e10
alpha1 = 1e-1
alpha2 = 3e-2
alpha3 = 6e-3
iter = 0
iter_list1 = []
delta_list1 = []
loss_list1 = []
iter_list2 = []
delta_list2 = []
loss_list2 = []
iter_list3 = []
delta_list3 = []
loss_list3 = []
# while delta > 1e-4:
while iter<1e3:
    new_x1 = my_x1 - alpha1*get_grad(my_x1)
    delta1 = np.linalg.norm(new_x1 - x_star, ord=2) / np.linalg.norm(x_star, ord=2)
    iter_list1.append(iter)
    delta_list1.append(np.log(delta1))
    loss_list1.append(get_loss(my_x1))
    print(f'{iter}, loss is {get_loss(my_x1)}, rela_x is {delta1}')
    my_x1 = new_x1

    new_x2 = my_x2 - alpha2 * get_grad(my_x2)
    delta2 = np.linalg.norm(new_x2 - x_star, ord=2) / np.linalg.norm(x_star, ord=2)
    iter_list2.append(iter)
    delta_list2.append(np.log(delta2))
    loss_list2.append(get_loss(my_x2))
    my_x2 = new_x2

    new_x3 = my_x3 - alpha3 * get_grad(my_x3)
    delta3 = np.linalg.norm(new_x3 - x_star, ord=2) / np.linalg.norm(x_star, ord=2)
    iter_list3.append(iter)
    delta_list3.append(np.log(delta3))
    loss_list3.append(get_loss(my_x3))
    my_x3 = new_x3

    iter += 1

# print(my_x1)
# print(my_x2)
# print(my_x3)
print(get_loss(my_x1))
print(get_loss(my_x2))
print(get_loss(my_x3))
# plt.plot(iter_list1, delta_list1, c='red', label='alpha=5e-1')
# plt.plot(iter_list2, delta_list2, c='blue', label='alpha=5e-2')
# plt.plot(iter_list3, delta_list3, c='green', label='alpha=5e-3')
plt.plot(iter_list1, loss_list1, c='red', label='alpha=5e-1')
plt.plot(iter_list2, loss_list2, c='blue', label='alpha=5e-2')
plt.plot(iter_list3, loss_list3, c='green', label='alpha=5e-3')
plt.xlabel('epoch')
plt.ylabel('f(x)')
plt.legend(loc=1)
plt.savefig('gd-loss.png', dpi=400)
