import pandas as pd
import numpy as np
from random import choice


class Worker:
    """
    A class representing workers, which is a collection of workers

    Each worker in the collection has heterogeneous reliability, and he has the form
    [c-i, d-i]
    """

    def __init__(self,filename, c0, d0):
        """

        :param filename: dataset filename(format ending should not be forgotten)
        :param c0: the initial c0 parameter for all the workers
        :param d0: the initial d0 parameter for all the workers
        """
        self._path = "D:\\task allocation\\noisy-oracle-master\\noisy-oracle-master\\data\\all_collected_data\\"
        self._filepath = self._path + filename
        # initialize the parameters of the workers prior distributions, generally c0 = 4, d0 = 1
        self._c0 = c0
        self._d0 = d0
        # the list that contains all the worker ids
        self._workers_id = []
        # the dict that contains all the workers prior distribution parameters. The key is worker id,
        # the value is the corresponding c-i and d-i
        self._workers_prior = {}
        # read the workers and initialize the workers prior distribution parameters
        self._initialize_worker_prior()

    def _initialize_worker_prior(self):
        """
        Create the self._worker_id list and initialize the worker prior distribution parameters
        :return: None
        """
        df = pd.read_csv(self._filepath, sep='	')
        self._workers_id = df['!amt_worker_ids'].unique().tolist()
        for worker_id in self._workers_id:
            self._workers_prior[worker_id] = [self._c0, self._d0]


    def get_worker_id_list(self):
        """
        Return the worker id list
        :return: self._workers_id
        """
        return self._workers_id


    def get_all_worker_prior(self):
        """
        Return the worker prior distribution parameter dictionary
        :return: self._workers_prior
        !!!Attention: simply return self._workers_prior means we get the real-time worker prior distribution dictionary!!!
        """
        return self._workers_prior

    def get_worker_prior(self,worker_id):
        """
        Return the corresponding worker prior distribution parameters
        :param worker_id: worker ID
        :return: corresponding worker prior distribution parameters
        """
        return self._workers_prior[worker_id]

    def update_parameter_c_d(self, wrk_id, a, b, c, d, z):
        """
        Update the worker prior distribution parameters using moment matching according to Appendix of the paper
        :param inst: the instance chosen at the stage t
        :param worker: the worker chosen at the stage t
        :param z: the label the worker gave to the instance
        :return: the worker's updated prior distribution parameters dict
        """
        # for key_, value_ in inst.items():
        #     a = value_[0]
        #     b = value_[1]
        # for key_, value_ in worker.items():
        #     c = value_[0]
        #     d = value_[1]
        if z == 1:
            exp_rho = c * (a * (c + 1) + b * d) / ((c + d + 1) * (a * c + b * d))
            exp_rho_square = c * (c + 1) * (a * (c + 2) + b * d) / ((c + d + 1) * (c + d + 2) * (a * c + b * d))
        elif z == 0:
            exp_rho = c * (b * (c + 1) + a * d)/((c + d + 1) * (b * c + a * d))
            exp_rho_square = c * (c + 1) * (b * (c + 2) + a * d)/((c + d + 1) * (c + d + 2) * (b * c + a * d))
        else:
            raise ValueError
        new_c = exp_rho * (exp_rho - exp_rho_square) / (exp_rho_square - np.square(exp_rho))
        new_d = (1 - exp_rho) * (exp_rho - exp_rho_square) / (exp_rho_square - np.square(exp_rho))

        # for key_ in worker.keys():
        self._workers_prior[wrk_id] = [new_c, new_d]
            # return self._workers_prior[key_]



def _test_workers():
    filename = 'rte.standardized.tsv'
    workers = Worker(filename, 4, 1)
    wk_id = workers.get_worker_id_list()
    wk_prior_dict = workers.get_all_worker_prior()
    inst = {'test':[1,1]}
    wk_id_choice = choice(wk_id)
    worker = {wk_id_choice:wk_prior_dict[wk_id_choice]}
    z = 1
    [c, d] = workers.update_parameter_c_d(inst, worker, z)
    wk_prior_dict_new = workers.get_all_worker_prior()

    print(wk_prior_dict)
    print(wk_prior_dict_new)


if __name__ == '__main__':
    _test_workers()
