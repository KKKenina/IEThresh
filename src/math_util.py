import numpy as np
from scipy.stats import beta
from numba import jit
import pickle



"""
Load the beta distribution dictionary from txt into python
:return: None
"""
try:
    with open('beta_dic.txt','rb') as f:
        beta_dic = pickle.load(f)
        if not isinstance(beta_dic, dict):
            beta_dic = {}
except EOFError:
    beta_dic = {}

print('beta_ab dictionary successful initialization')


@jit
def new_a_b(a, b, c, d, z):
    """
    Calculate the new a and new b when traversing all the available instances and the corresponding workers
    based on a, b, c, d at present
    :param a: current a-i at stage t, used to get new a, b given label z
    :param b: current b-i at stage t, used to get new a, b given label z
    :param c: current c-i at stage t, used to get new a, b given label z
    :param d: current d-i at stage t, used to get new a, b given label z
    :param z: label z
    :return: corresponding new a and new b
    """
    if z == 1:
        exp_theta = a * ((a + 1) * c + b * d) / ((a + b + 1) * (a * c + b * d))
        exp_theta_square = a * (a + 1) * ((a + 2) * c + b * d) / ((a + b + 1) * (a + b + 2) * (a * c + b * d))
    elif z == 0:         # the RTE dataset have two label results : 0 and 1, different from the paper
        exp_theta = a * (b * c + (a + 1) * d) / ((a + b + 1) * (b * c + a * d))
        exp_theta_square = a * (a + 1) * (b * c + (a + 2) * d) / ((a + b + 1) * (a + b + 2) * (b * c + a * d))
    else:
        raise ValueError
    new_a = exp_theta * (exp_theta - exp_theta_square) / (exp_theta_square - np.square(exp_theta))
    new_b = (1 - exp_theta) * (exp_theta - exp_theta_square)/(exp_theta_square - np.square(exp_theta))

    return new_a, new_b

@jit
def Beta_ab_cdf(a, b):
    """
    calculate I(a,b) defined in the paper, using the function beta.sf
    :param a: beta distribution parameter a
    :param b: beta distribution parameter b
    :return: Pr( theta > 0.5 | theta ~ Beta(a, b) )
    """

    if beta_dic.get((a, b)) == None:
        beta_dic[(a, b)] = beta.sf(0.5,a, b)
    I_ab = beta_dic[(a, b)]
    # a = float(a)
    # b = float(b)
    # I_ab = 1 - eng.cdf('Beta', 0.5, a, b)
    # I_ab = 0.3
    # I_ab = beta.sf(0.5, a, b)

    return I_ab


def save_beta_dic():
    """
    Save the beta distribution dictionary to 'beta_dic.txt' using pickle
    :return: None
    """
    f = open('beta_dic.txt','wb')
    pickle.dump(beta_dic, f)
    f.close()


def load_beta_dic():
    """
    Load the beta distribution dictionary from txt into python
    :return: None
    """
    try:
        with open('beta_dic.txt','rb') as f:
            _beta_dic = pickle.load(f)
            if not isinstance(_beta_dic, dict):
                _beta_dic = {}
    except EOFError:
        _beta_dic = {}

    beta_dic = _beta_dic



@jit
def h_function(I):
    """
    h function as defined in the paper
    h(x) = max(x, 1-x)
    :param I: x of the function
    :return: comparison result, return the comparatively bigger one
    """
    result = max(I , (1 - I))
    return result


def _test_math_util():
    load_beta_dic()
    a = 1
    b = 1
    I_ab = Beta_ab_cdf(a, b)
    print(I_ab)



if __name__ == '__main__':
    _test_math_util()