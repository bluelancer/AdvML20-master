import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

class VI:
    def __init__(self):
        self.a_0 = 0
        self.b_0 = 0
        self.mu_0 = 0
        self.lambda_0 = 0
        self.iteration = 10
        self.N = 10 #
        self.data = self.get_gaussian(self.N, 0, 0.5)

    # Data Generation
    def get_gaussian(self, N, mu, sigma, seed = 10):
        np.random.seed(seed)
        GD = np.random.normal(mu,sigma, N)
        return GD

    # Probs
    def E_tau(self, a,b):
        return a/(b + 0.000001) # avoid divided by 0

    def E_mu (self, mu_N):
        return mu_N

    def E_mu_square (self, lambda_N, mu_N):
        return self.E_mu(mu_N) ** 2 + 1/(lambda_N + 0.000001)  # avoid divided by 0

    # Updates
    def update_a(self, a_0, N):
        return a_0 + N/2 # a_N = a_0 + N/2
    def update_b(self, b_0, lambda_0, data, lambda_N,mu_0, mu_N):
        xmu_sum = np.sum(data**2) -2 * mu_N* np.sum(data) + len(data) * self.E_mu_square(lambda_N,mu_N)
        lambdamu_sum = lambda_0 * (self.E_mu_square(lambda_N,mu_N) -2* mu_0* mu_N + mu_0**2)
        b_N = b_0 + 0.5 * (xmu_sum + lambdamu_sum)
        return b_N # b_0 + 0.5E_mu[\sum (x_n-mu)^2 + \lambda_0(mu-mu_0)^2]
    def update_mu(self, lambda_0, mu_0, N, data):
        return (lambda_0 * mu_0 + N * data.mean())/(lambda_0 + N) # mu_N = lambda_0* mu_0 + N* x.mean / lambda_0 +N
    def update_lambda(self,lambda_0,a_n,b_n,N):
        return (lambda_0 + N) * self.E_tau(a_n,b_n) # (lambda_0 + N)E(tau)

    # Training
    def training(self):
        self.a_N = self.update_a(self.a_0,self.N)
        self.mu_N = self.update_mu(self.lambda_0, self.mu_0, self.N, self.data)
        self.lambda_N = self.update_lambda(self.lambda_0,self.a_0,self.b_0,self.N)
        self.b_N = self.update_b(self.b_0, self.lambda_0,self.data,self.lambda_N,self.mu_0,self.mu_N)
        # print(gamma(self.a_N))
        for i in range(self.iteration):
            self.b_N = self.update_b(self.b_0, self.lambda_0, self.data, self.lambda_N, self.mu_0, self.mu_N)
            self.lambda_N = self.update_lambda(self.lambda_0, self.a_N, self.b_N, self.N)
            # print ("b_N = ", self.b_N)
            # print ("lambda_N = ",self.lambda_N)

    #Q = q_μ * q_τ
    #q_μ(μ) = N(μ|μ_N,λ^−1_N)
    def get_q_mu_optim (self, mu_N, lambda_N, mu):
        q_mu_optim = np.sqrt(lambda_N/(2*np.pi)) *\
                     np.exp(-0.5 *np.dot(lambda_N,np.transpose(mu - mu_N)**2))
        return q_mu_optim

    #Gam(τ |aN, bN)
    def get_q_tau_optim(self, a_N, b_N, tau):
        q_tau_optim = (1/gamma(a_N)) * b_N**a_N * tau**(a_N-1) *np.exp(-b_N * tau)
        return q_tau_optim

    def exact_posterior(self, lambda_0, mu_0, a_0, b_0, mu, tau):
        gd_component = np.sqrt(lambda_0 * tau / (2 * np.pi)) * np.exp(-0.5 * np.dot(lambda_0 * tau, ((mu - mu_0) ** 2)))
        gamma_component = ((b_0 ** a_0) / gamma(a_0)) * (tau ** (a_0 - 0.5)) * (np.exp(-b_0*tau))
        posterior = gd_component.transpose() * gamma_component
        return posterior

    def inference(self,a_N ,b_N,lambda_N,mu_N):
        mu = np.linspace(-1,1, num=100)
        tau = np.linspace(0,2, num=100)

        mu_polt,tau_polt = np.meshgrid (mu, tau)

        q_mu = self.get_q_mu_optim(mu_N, lambda_N, mu_polt)
        q_tau = self.get_q_tau_optim(a_N, b_N, tau_polt)
        q_approx = q_mu * q_tau.transpose()

        posterior = self.exact_posterior(self.N, self.data.mean(), 0.5 * self.N, 0.5 * np.sum(self.data - self.data.mean()), mu_polt, tau_polt)

        # print(posterior.size)
        plt.figure()
        plt.contour(mu_polt,tau_polt,q_approx,colors = "b")
        plt.contour(mu_polt,tau_polt,posterior, colors = "r")
        plt.savefig("2_3")

        return q_approx, posterior

if __name__ == '__main__':
    vi = VI()
    vi.training()
    vi.inference(vi.a_N,vi.b_N,vi.lambda_N,vi.mu_N)




