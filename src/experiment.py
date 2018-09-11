import numpy as np
# show the figure opt-KG on RTE
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import warnings
from dataset import DataSource
from workers import Worker
from math_util import *
from algorithm import Algorithm


#ignore the unnecessary warnings
warnings.filterwarnings('ignore')


def Opt_KG_experiment():
    """
    Run experiment on the RTE data set
    :return: accuracy
    """
    # initialize the prior parameters a0 b0 of the dataset
    data_file = 'rte.standardized.tsv'
    init_a0 = 1
    init_b0 = 1
    # sourcedata = DataSource(data_file, init_a0, init_b0)
    # initialize the prior parameters c0 d0 of the workers
    init_c0 = 4
    init_d0 = 1
    # workers = Worker(data_file, init_c0, init_d0)
    # Given Budget T
    Budget_T = np.arange(0,8000,100)
    # accuracy result of experiment each time
    accuracy_ = []
    # run experiment limited to the given budget T_
    for T_ in Budget_T:
        accuracy_sum = 0
        for i in range(0,1):
            sourcedata = DataSource(data_file, init_a0, init_b0)
            workers = Worker(data_file, init_c0, init_d0)
            Opt_KG = Algorithm(sourcedata, workers, T_)
            H_T, H_complement = Opt_KG.run_Opt_KG()
            # the number that the positive and negative set result derived from the experiment is accordance with the real data
            result = 0
            # get H* and H*c
            H_star, H_star_c = sourcedata.get_H_star()
            for idx in H_T:
                if idx in H_star:
                    result = result + 1
            for idx in H_complement:
                if idx in H_star_c:
                    result = result + 1
            # calculate the accuracy_sum
            accuracy_sum = accuracy_sum + result / 800
        # calculate the accuracy
        accuracy_mean = accuracy_sum / 1
        accuracy_.append(accuracy_mean)
        # print the accuracy result on the console
        print('the length of H_t is:' + str(len(H_T)) + ', the length of H_t_c is:' + str(len(H_complement)))
        print('the length of H* is:' + str(len(H_star)) + ', the length of H*_c is:' + str(len(H_star_c)))
        print('the length of result is:' + str(result))
        print('Budget ' + str(T_) + ' and the accuracy is ' + str(accuracy_[-1]))
        print('*' * 40)


    # save the beta distribution dictionary
    save_beta_dic()
    # plot
    plt.figure()
    plt.plot(Budget_T, accuracy_)
    plt.xlabel('Budget')
    plt.ylabel('accuracy')
    plt.title('Opt-KG on RTE')
    plt.show()



if __name__ == '__main__':
    Opt_KG_experiment()