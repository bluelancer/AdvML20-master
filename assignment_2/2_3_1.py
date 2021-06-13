import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Data Generation
def get_data(N, mu, sigma, seed = 10):
    np.random.seed(seed)
    GD = np.random.normal(mu,sigma, N)
    return GD

def posterior( mu_0,lambda_0,a_0, b_0, mu, tau):
    return  tau ** (a_0 -1/2)*np.exp(-tau * b_0) * np.exp(-tau*(lambda_0/2)*(mu-mu_0)**2)

if __name__ == '__main__':
    a_0 = 0
    b_0 = 100
    mu_0 = 0
    lambda_0 = 1
    iter = 1000
    N = 100
    data = get_data(N, 2, 1)
    thresh = 1e-8

    a_star = a_0 + N/2  # a_star = a0+(N+1)/2

    # mu_star = lambda_0* mu_0 + N* x.mean / lambda_0 +N
    mu_star = (lambda_0 * mu_0 + N * np.mean(data))/(lambda_0 + N)
    lambda_star = (lambda_0 + N) * a_0/(b_0+1e-9)
    sum_data_2 = np.sum(data ** 2)
    sum_data = np.sum(data)
    x_mu_sum = sum_data_2 - 2 * mu_star * sum_data + len(data) * (mu_star ** 2 + 1 / (lambda_star +1e-9))
    lambda_mu_sum = lambda_0 * ((mu_star ** 2 + 1 / (lambda_star + 1e-9)) - 2 * mu_0 * mu_star + mu_0 ** 2)
    b_star  = b_0 + 0.5 * (x_mu_sum + lambda_mu_sum)

    lambda_prev = lambda_0
    b_prev = b_0
    # print(gamma(self.a_star))

    for i in range(iter):
        if (abs(lambda_star - lambda_prev) > thresh) or (abs(b_star - b_prev) > thresh):
            lambda_prev = lambda_star
            b_prev = b_star
            sum_data_2 = np.sum(data ** 2)
            sum_data = np.sum(data)
            x_mu_sum = sum_data_2 - 2 * mu_star * sum_data + len(data) * (mu_star ** 2 + 1 / (lambda_star + 1e-9))
            lambda_mu_sum = lambda_0 * ((mu_star ** 2 + 1 / (lambda_star + 1e-9)) - 2 * mu_0 * mu_star + mu_0 ** 2)
            b_star = b_0 + 0.5 * (x_mu_sum + lambda_mu_sum)
            lambda_star = (lambda_0 + N) * a_star/(b_star+1e-9)
        else:
            break

    mu = np.linspace(1, 3, num=100)
    tau = np.linspace(0, 2, num=100)

    mu_list, tau_list = np.meshgrid(mu, tau)

    q_mu = np.sqrt(lambda_star/(2*np.pi)) *\
                 np.exp(-0.5 *np.dot(lambda_star,np.transpose(mu - mu_star)**2))
    q_tau  = (1/gamma(a_star)) * b_star**a_star * tau_list**(a_star-1) *np.exp(-b_star * tau_list)
    q_approx = q_mu.transpose() * q_tau

    posterior = posterior(
        (lambda_0 * mu_0 + N * np.mean(data)) / (lambda_0 + N),N + lambda_0, 0.5 * N + a_0,
        b_0 + 0.5 * (N * np.var(data) + (lambda_0 * N * (np.mean(data) - mu_0) ** 2) / (lambda_0 + N)), mu_list, tau_list)

    # print(posterior.size)
    plt.figure()
    plt.contour(mu_list, tau_list, q_approx, colors="b")
    plt.contour(mu_list, tau_list, posterior, colors="r")
    plt.savefig("2_3_2")

