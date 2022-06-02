# Solves the LSM algorithm by Longstaff and Schwartz(2001)

#import packages
import pandas as pd
import numpy as np
import chaospy # Used to obtain Halton draws
import time
from statistics import NormalDist

def run_simulations(S0, strike, T, M, r, sigma, simulations, deg, basis_func, method, halton = None, reference_value = None):
    """ Simulates the price of an American Option using the Option pricing class:
    Parameters
    ----------
    S0: float : The initial price of the underlying asset.
    strike: float : the strike price of the option.
    T: float : time to maturity (years)
    M: int : number of time steps per year
    r: float : constant risk-free interest rate
    sigma : float : constant volatility 
    simulations: int: number of simulated price paths
    basis_func : str: selected basis function, currently implemented: polyfit, chebyshev, laguerre, hermite, legendre
    deg: int: number of polynomial terms
    method: str: selected method, currently implemented: 'monte carlo' or 'brownian bridge' 
    halton: boolean: if true then halton draws are used
            
    Returns
    =======
    price: ndarray : price of the American option
    time_elapsed: ndarray : time used to simulate the price 
    standard_dev: ndarray : standard deviation
    standard_error: ndarray : standard error
    relative_error: ndarray : relative error
    """
     # start timer
    t0 = time.time() 

    # initialize class
    PUT = OptionPricingClass(S0, strike, T, M, r, sigma, simulations, deg, basis_func, method, halton)

    # price option
    price, standard_dev, standard_error = PUT.price

    # relative error
    if reference_value == None:
        relative_error = np.nan
    else:
        relative_error = (price-reference_value)**2

    # stop timer
    t1 = time.time()
    time_elapsed = t1-t0

    return (price, time_elapsed, standard_dev, standard_error, relative_error)


#### Used for Brownian Bridge ####
map_to_norm = lambda u: NormalDist(mu=0, sigma=1).inv_cdf(u) # Standard normal quantile function 
map_to_norm_f = np.vectorize(map_to_norm) # Make map_to_norm take vector input and return vector

def halton_draws(n):
    """ Gets Halton Draws from a Normal distribution, with default base 2.
    ----------
    n: float : number of draws
    
    Returns
    =======
    draws: ndarray : Halton draws
    """
    nodes = chaospy.create_halton_samples(n, dim=1, burnin=- 1).flatten()
    # Storage for uniform draws within Halton node bins
    draws = map_to_norm_f(nodes)
    return draws
#### Used for Brownian Bridge ####

# Class for pricing American put option: Solves the LSM algorithm by Longstaff and Schwartz(2001)
class OptionPricingClass(object):
    """ Class for American put option pricing using Longstaff-Schwartz (2001):
    Parameters
    ----------
    S0: float : initial stock price
    strike: float : the strike price of the option
    T: float : time to maturity (years)
    M: int : time steps per year
    r: float : constant risk-free rate
    sigma : float : constant volatility 
    S0: float : initial stock price
    strike: float : the strike price of the option
    simulations: int: number of simulated price paths
    basis_func : str: selected basis function, currently implemented: polyfit, chebyshev, laguerre, hermite, legendre
    deg: int: number of degrees in polynomial 
    method: str: selected method, currently implemented: 'monte carlo' or 'brownian bridge' 
    halton: boolean: if true then halton draws are used
    
    """

    def __init__(self, S0, strike, T, M, r, sigma, simulations, deg, basis_func, method, halton = False):
        self.S0 = float(S0)
        self.strike = float(strike)
        self.T = float(T)
        self.M = int(M)
        self.r = float(r)
        self.sigma = float(sigma)
        self.simulations = int(simulations)
        self.deg = int(deg)
        self.basis_func = basis_func
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)
        self.method = method
        self.halton = halton
        
        # fixed
        self.seed = 123


    @property
    def MCprice_matrix(self):
        """Returns Monte Carlo price matrix rows: time columns: price-path simulation """
        np.random.seed(self.seed)
        
        # Initialize
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0,:] = self.S0 
        
        # Forward simulation
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal(int(self.simulations / 2)) # randome draws
            brownian = np.concatenate((brownian, -brownian)) # antithetic draws
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
                                  + self.sigma * brownian * np.sqrt(self.time_unit))) # Black and Scholes price paradigme
        return MCprice_matrix

    @property
    def MCpayoff(self):
        """Returns the inner-value of American Put Option"""
        payoff = np.maximum(self.strike - self.MCprice_matrix,
                            np.zeros((self.M + 1, self.simulations),
                            dtype=np.float64))
        return payoff

    @property
    def MCcashflow_vector(self):
        """Returns the inner-value of American Put Option Using LSM"""  
        
        # Initialize
        cashflow_vec = np.zeros_like(self.MCpayoff)
        cashflow_vec[-1, :] = self.MCpayoff[-1, :]
        ITM_index = self.MCpayoff > 0 #The In The Money index
        
        # Backward induction to solve for the optimal stopping rule
        for t in range(self.M - 1, 0 , -1):
            ITM_t = ITM_index[t] 
            price_t = self.MCprice_matrix[t, :]
            value_t = cashflow_vec[t + 1, :] * self.discount
            exercise_t = self.MCpayoff[t, :] # The immediate exercise value is the payoff
            
            # Only the ITM paths are evaluated
            exercise_itm = exercise_t[ITM_t]  
            X = np.array(price_t[ITM_t]) # In the money price matrix at time t
            Y = np.array(value_t[ITM_t]) # In the money cash-flow matrix at time t
            
            ### Update cash-flow matrix 
            if len(X) == 0:
                cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount #If none of the cash-flow are in the money, set the cash-flows equal to the discounted future value
            else:
                
            # Selected Basis Function
                if self.basis_func == 'polyfit': 
                    regression = np.polyfit(X, Y, deg = self.deg) # Fit Regression
                    continuation_value = np.polyval(regression, X) # Continuation Values 
                    cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount
                    cashflow_vec[t, :][ITM_t] = np.where(exercise_itm > continuation_value, exercise_itm,
                                              cashflow_vec[t + 1, :][ITM_t] * self.discount) # Update cash-flow  

                if self.basis_func == 'chebyshev': 
                    reg = np.polynomial.chebyshev.chebfit(X, Y, deg= self.deg, rcond=None, full=False, w=None) # fit regression
                    continuation_value = np.polynomial.chebyshev.chebval(X, reg) # continuation values
                    cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount # The OTM cash-flows are set to the discounted cash flows
                    cashflow_vec[t, :][ITM_t] = np.where(exercise_itm > continuation_value, exercise_itm,
                                              cashflow_vec[t + 1, :][ITM_t] * self.discount) # Update cash-flow  


                if self.basis_func == 'laguerre': 
                    reg = np.polynomial.laguerre.lagfit(X, Y, deg= self.deg, rcond=None, full=False, w=None) # fit regression
                    continuation_value = np.polynomial.laguerre.lagval(X, reg) # continuation values
                    cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount
                    cashflow_vec[t, :][ITM_t] = np.where(exercise_itm > continuation_value, exercise_itm,
                                              cashflow_vec[t + 1, :][ITM_t] * self.discount) # Update cash-flow  


                if self.basis_func == 'hermite': 
                    reg = np.polynomial.hermite.hermfit(X, Y, deg= self.deg, rcond=None, full=False, w=None) # fit regression
                    continuation_value = np.polynomial.hermite.hermval(X, reg) # continuation values
                    cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount
                    cashflow_vec[t, :][ITM_t] = np.where(exercise_itm > continuation_value, exercise_itm,
                                              cashflow_vec[t + 1, :][ITM_t] * self.discount) # Update cash-flow 


                if self.basis_func == 'legendre': 
                    reg = np.polynomial.legendre.legfit(X, Y, deg= self.deg, rcond=None, full=False, w=None) # fit regression
                    continuation_value = np.polynomial.legendre.legval(X, reg) # continuation values
                    cashflow_vec[t, :] = cashflow_vec[t + 1, :] * self.discount
                    cashflow_vec[t, :][ITM_t] = np.where(exercise_itm > continuation_value, exercise_itm,
                                              cashflow_vec[t + 1, :][ITM_t] * self.discount) # Update cash-flow 
                
                
        return cashflow_vec[1,:] * self.discount
    
    @property
    def BBcashflow_vector(self):
        """Returns the inner-value of American Put Option Using LS-BB"""  
        # set seed
        np.random.seed(self.seed)

        # Parameters 
        dt = self.time_unit 

        # Initial and final draw
        W_0 = 0

        if self.halton:
            W = halton_draws(self.simulations)
        else:
            W = np.random.normal(0, 1, size = self.simulations)

        # Initial and final stock price
        S_0 = np.empty(shape=(self.simulations)) # one column per sim
        S_0.fill(self.S0)
        S_T = S_0 * np.exp((self.r - 1/2 * self.sigma**2)*self.T + np.sqrt(self.T) * self.sigma * W)

        # Initial and final 
        cashflow_vec = np.fmax(0,self.strike - S_T).flatten() * np.exp(-self.r*dt)

        # backward induction
        for t in range(self.M-1, -1, -1):
            # Generate new draw
            eps = np.random.normal(0, 1, size = (1, self.simulations))
            W_t = t/(t+1) * W + np.sqrt(t/(t+1)*dt) * eps
            dW_t = W_t - W_0

            # Generate current stock price
            S_t = S_0 * np.exp((self.r - 1/2 * self.sigma**2)*t*dt + self.sigma * dW_t)

            # Exercise payoffs
            pay_off_t = np.fmax(0, self.strike - S_t).flatten()
            
            # ITM index
            ITM_index = np.where(pay_off_t > 0)[0]

            if len(ITM_index) == 0:
                cashflow_vec = cashflow_vec * np.exp(-self.r*dt)
            else:
                X = S_t[:,ITM_index].flatten()
                Y = cashflow_vec[ITM_index]

                # fit regression
                reg = np.polynomial.laguerre.lagfit(X, Y, deg = self.deg, rcond=None, full=False, w=None) 
                
                # continuation values
                Continuation = np.polynomial.laguerre.lagval(X, reg).reshape(1,-1).flatten() 
                
                # favorable continuation index
                Execute_index = ITM_index[pay_off_t[ITM_index] > Continuation]

                cashflow_vec[Execute_index] = pay_off_t[Execute_index]

                if t != 0:
                    cashflow_vec = cashflow_vec * np.exp(-self.r*dt)
                else:
                    cashflow_vec = cashflow_vec

                # save W for next iteration
                W = W_t
                
        return cashflow_vec
    

    @property
    def price(self):
        """Returns the Price of American Put Option"""  

        if self.method == 'monte carlo':
            cashflow_vector = self.MCcashflow_vector
        if self.method == 'brownian bridge':
            cashflow_vector = self.BBcashflow_vector
            
        price = np.sum(cashflow_vector) / float(self.simulations)
        std_dev = np.sqrt(np.sum((cashflow_vector - price)**2)/float(self.simulations-1)) # standard deviation
        se = std_dev / np.sqrt(float(self.simulations)) # standard error    
        
        return price, std_dev, se  
    
    
    
    
