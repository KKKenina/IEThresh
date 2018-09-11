import numpy as np
import pandas as pd
import matplotlib as plt
from random import choice



class DataSource:
    """
    A class representing the whole data set, which is a collection of instances derived from RTE data set.

    Each instance in the data set is uniquely distinguished by the "orig_id", and has a prior distribution
    at any stage t ( 0<= t <= T-1)
    """
    def __init__(self, filename, a0, b0):
        self._path = "D:\\task allocation\\noisy-oracle-master\\noisy-oracle-master\\data\\all_collected_data\\"
        self._filepath = self._path + filename
        self._dataset = {}
        self._inst_id = []
        # read the data from the given file
        self._read_data_from_file()
        # parameters of prior distributions for instances and for workers
        self._a0 = a0
        self._b0 = b0
        # parameters of prior distributions for instances
        self._inst_prior = {}
        # initialize each instance's prior distribution
        self._initialize_prior_distribution()



    def _read_data_from_file(self):
        """
        Create the data set from the given file so that self._dataset is a dict of instances, where each item
        in the dict is of the form {"orig_id":content}
        :return: None
        """
        df = pd.read_csv(self._filepath, sep='	')
        task_worker = df.groupby('orig_id')['!amt_worker_ids'].apply(lambda x: x.tolist()).to_dict()
        task_response = df.groupby('orig_id')['response'].apply(lambda x: x.tolist()).to_dict()
        task_gold = df.groupby('orig_id')['gold'].apply(lambda x: x.unique()).to_dict()

        for key_, value_ in task_worker.items():
            content = {}
            content['workers'] = value_
            content['response'] = task_response[key_]
            content['gold'] = task_gold[key_]
            self._dataset[key_] = content
            self._inst_id.append(key_)


    def _initialize_prior_distribution(self):
        """
        Initialize the parameters of the instances prior distributions
        We use the uniform prior Beta(1,1) for each theta-i, in other words, the value of the parameters a-i
        and b-i is 1.
        :return:
        """
        for task_idx in self._dataset.keys():
            inst_prior_ = [self._a0, self._b0]
            self._inst_prior[task_idx] = inst_prior_



    def update_parameter_a_b(self, task_id, a, b, c, d, z):
        """
        Update the parameters of the instance prior distributions using moment matching according to the Appendix
        :param inst: the instance chosen at the stage t, and the inst is a item of self._inst_prior
        :param worker: the worker chosen at the stage t
        :param z: the label worker j annotated to the instance i at stage t
        :return: None
        """
        # for key_,value_ in inst.items():
        #     a = value_[0]
        #     b = value_[1]
        # for key_, value_ in worker.items():
        #     c = value_[0]
        #     d = value_[1]
        if z == 1:
            exp_theta = a *((a + 1) * c + b * d) / ((a + b + 1) * (a * c + b * d))
            exp_theta_square = a *(a + 1) * ((a + 2) * c + b * d) / ((a + b + 1) * (a + b + 2) * (a * c + b * d))
        elif z == 0:         # the RTE dataset have two label results : 0 and 1, different from the paper
            exp_theta = a * (b * c + (a + 1) * d) / ((a + b + 1) * (b * c + a * d))
            exp_theta_square = a * (a + 1) * (b * c + (a + 2) * d) / ((a + b + 1) * (a + b + 2) * (b * c + a * d))
        else:
            raise ValueError

        new_a = exp_theta * (exp_theta - exp_theta_square) / (exp_theta_square - np.square(exp_theta))
        new_b = (1 - exp_theta) * (exp_theta - exp_theta_square)/(exp_theta_square - np.square(exp_theta))

        # for key_ in inst.keys():
        self._inst_prior[task_id] = [new_a, new_b]
            # return self._inst_prior[key_]


    def get_inst_prior_parameter(self, inst_id):
        """
        Get the parameters of the instance prior distribution
        :param inst_id: "org_id",the instance
        :return: a list which contains a-i and b-i of the instance i
        """
        a = self._inst_prior[inst_id][0]
        b = self._inst_prior[inst_id][1]
        li = [a, b]
        prior_distribution = {inst_id: li}
        return prior_distribution


    def get_dataset(self):
        """
        Return the dataset of the dict form
        :return: self._dataset
        """
        return self._dataset


    def get_inst_id_list(self):
        """
        Get a list that contains all the ids of the instances
        :return: a list containing instances ids
        """
        return self._inst_id


    def get_all_inst_prior_distribution(self):
        """
        Return the current dict of the parameters of instances prior distribution
        :return: self._inst_prior
        """
        return self._inst_prior


    def get_H_star(self):
        """
        Return the set H* and H*c i.e. the set containing the instances whose gold answer are 1
        :return:H_star, H_star_c
        """
        H_star = []
        H_star_c = []
        for key_, value_ in self._dataset.items():
            if value_['gold'] == 1:
                H_star.append(key_)
            if value_['gold'] == 0:
                H_star_c.append(key_)

        return H_star, H_star_c



#     test the class DataSource
def _test_dataset():
    filename = "rte.standardized.tsv"
    a0 = 1
    b0 = 1
    datasource = DataSource(filename, a0, b0)
    datadict = datasource.get_dataset()
    H_star, H_star_c = datasource.get_H_star()
    inst_id = choice(datasource.get_inst_id_list())
    z = choice(datadict[inst_id]['response'])
    inst_prior = datasource.get_inst_prior_parameter(inst_id)
    worker = {'test':[4,1]}
    updated = datasource.update_parameter_a_b(inst_prior, worker, z)
    inst_prior = datasource.get_all_inst_prior_distribution()
    print(updated)
    print(inst_prior)



if __name__ == "__main__":
    _test_dataset()




