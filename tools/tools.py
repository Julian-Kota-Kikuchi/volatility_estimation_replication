import numpy as np
from scipy import integrate
from scipy.special import erfc, erfinv, gamma, gammainc
from scipy.stats import norm

# speed - up tools
from numba import int64, float64, jit
from functools import lru_cache

@jit(nopython=True)
def delta_n_vector(t,  delta):
    
    delta_n = np.zeros_like(t)

    for i in range(t.shape[0]):
        if np.abs(t[i]) <= 1 / delta:
            delta_n[i] = delta * (1 - delta * np.abs(t[i]))
    
    return delta_n


@jit(float64(float64, float64), nopython=True, cache=True)
def delta_n(t, delta):
    '''
    Delta function as defined in (5.2) in Watson and Leadbetter (1964)
    
    Parameters
    ----------
    t : float64
        t - t_g,i ^ W
    n : float64
        number of 
        
    Returns
    -------
    float64
        returns the output of the delta function
    '''
    
    if np.absolute(t) <= 1/delta:  

        delta_n = delta*(1 - delta*np.absolute(t))

    else:
         delta_n = 0
    
    return delta_n


@jit(nopython=True)
def f_hat_t(t, tau, delta):
    '''
    Function to estimate pdf.
    
    Parameters
    ----------
    t : float64
        t - t_g,i ^ W
    tau : float64
        tau value
    n : float64
        number of iterations
    
    Returns
    -------
    float64
        returns estimate of the pdf at a particular t
    '''
    
    delta_s = delta_n_vector(t - tau, delta)
    f_hat_t = np.nanmean(delta_s)
    
    return f_hat_t


# function for F_hat
def F_hat_t(t, tau, delta): 
    '''
    Function to estimate cdf.
    
    Parameters
    ----------
    t : float64
        t - t_g,i ^ W
    delta_function : function
        dirtac delta-like function
    tau : ndarray
        containint data with type flaot
    n : float64
        number of iterations
    Returns
    -------
    ndarray
        returns estimate of the cdf at a particular t
    '''
    sample_size = len(tau)
    
    if type(t) == int or type(t)== np.float64 or float:
        t = np.tile(t, sample_size+1)
    
    F_hat_t = np.mean([u_n(t[i] - tau[i], delta) for i in range(0, sample_size)])
    
    return F_hat_t

#@jit(nopython=True)
def u_n(t, delta):
    '''
    Function to intergrate delta function as deined in Watson and Leadbetter (1964).
    
    Parameters
    ----------
    t : float64
        t - t_g,i ^ W
    delta_function : function
        dirtac delta-like function
    n : float64
        sample size
    
    Returns
    -------
    float64
        returns result of integration
    '''
    
    if t < -1/delta:
        return 0
    
    elif -1/delta <= t < 0: 
        return delta*t + (1/2)*(delta**2)*t**2 + 1/2
    
    elif 0 <= t < 1/delta:
        return 0.5 + delta*t - (1/2)*(delta**2)*t**2
    
    else:
        return 1

    
def a_n(t, delta_function, delta):
    '''
    Function to intergrate delta function as deined in Watson and Leadbetter (1964).
    
    Parameters
    ----------
    t : float64
        t - t_g,i ^ W
    delta_function : function
        dirtac delta-like function
    n : float64
        sample size
    
    Returns
    -------
    float64
        returns result of integration
    '''
    
    def f(t):
        return delta_function(t, delta)**2

    return integrate.quad(f, -np.inf, 0, limit = 30000)[0] + integrate.quad(f, 0, np.inf, limit = 30000)[0]


def fgW(t, g):
    return g/np.sqrt(2*np.pi*t**3)*(np.exp(-g**2/(2*t)))

def PgW(t, g):
    return 1 - erfc(g/np.sqrt(2*t))
    
def PgW_inv(t, g):
    return g**2/(2*erfinv(1 - t)**2)

def sigma_hat_t(f_hat, F_hat, g):
    '''
    Function to estimate spot volatility.
    
    Parameters
    ----------
    f_hat : float64
        estimated pdf
    F_hat : float64
        estimated cdf
    
    Returns
    -------
    float64
        returns estimate of spot volatility
    '''
    ret = []
    
    if (type(f_hat) != list) & (type(f_hat) != np.ndarray):
        f_hat = np.array([f_hat])
        
    if (type(F_hat) != list) & (type(F_hat) != np.ndarray):
        F_hat = np.array([F_hat])
        
    
    for i in range(len(f_hat)):
        if (F_hat[i]==0 or F_hat[i]==1):
            ret.append(0)
        else:
             ret.append(f_hat[i]/fgW(PgW_inv(F_hat[i], g), g))
    
    return ret


def s(n, a_n, f_g_w, PgW_inv, F_t, f_t, sigma_hat, sigma):
    '''
    Function calculate s_1 as defined in 6.1.
    
    Parameters
    ----------
    n : float64
        sample size
    a_n : float64
    f_g_w : function
    PgW_inv : function
    F_t : float64
        true cdf value at t
    f_t : float64
        true pdf value at t
    sigma_hat: float64
        estimated s_t
    sigma: float64
        true s_t
    
    Returns
    -------
    float64
        returns result of the calculation of s
    '''
    if type(sigma) == list:
        sigma = np.array(sigma_hat)
    
    first_part = n/a_n
    
    middle = ((f_g_w(PgW_inv(F_t)))**2)/f_t
    
    last = sigma_hat - sigma
    
    
    return np.sqrt(first_part*middle)*last


# function to summarize estimation
def estimation_summary(estimate, summary_statistics, theoretical_values = None, rounding = None):
    summary = []
    
    if rounding:
        rounds = rounding
    else:
        rounds = 5
        
    if type(estimate) == list:
        estimate = np.array(estimate)
    
    for i in summary_statistics:
        if type(i) == int or type(i) ==  float:
            summary.append(np.nanquantile(estimate, i).round(rounds))
            
        elif i in [np.mean, np.std, np.nanmean, np.nanstd, np.nanmedian, np.nanmax, np.nanmin]:
            summary.append(round(i(estimate),rounds))
            
        elif i in ['rmse']:
            summary.append(np.sqrt(round(np.nanmean(np.array(theoretical_values) - estimate)**2, rounds)))
        
        elif i in ['mse']:
            summary.append(round(np.nanmean(np.array(theoretical_values) - estimate)**2, rounds))
        
        elif i in ['bias']:
            summary.append(round(np.mean(estimate - theoretical_values),rounds))
            
        elif i in ['n']:
            summary.append(len(estimate))
            
        elif i in ['true']:
            summary.append(theoretical_values)
            
    
    return summary