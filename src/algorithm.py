import numpy as np
from dataset import DataSource
from math_util import *
from workers import Worker
import matplotlib.pyplot as plt
import copy
class Algorithm:
    """
    The algorithms presented in the paper 'Statistical Decision Making for Optimal Budget Allocation in Crowd Labeling'
    The class is named as Algorithm since I at first want to code all possible algorithms in the paper so that comparison
    experiments can be done
    Currently this class only contains one algorithm:Opt-KG
    """

    def __init__(self, instances, workers, budget):
        """
        Initialize the experiment for the given instances, the workers and the buduget T
        :param instances: the given dataset (of type DataSource)
        :param workers: the workers (of type Worker)
        :param budget: the given experiment budget T
        """
        self._instances = instances
        self._workers = workers
        self._param_of_all_insts = self._instances.get_all_inst_prior_distribution()
        self._param_of_all_wrks = self._workers.get_all_worker_prior()
        self._T = budget
        # traversing available instances
        self._instances_ramain = {}
        # The list containing all the indexes of the instances whose a > b
        self._positive_set = []
        # instances that have been chosen
        self._chosen_inst = []


    def _initialize_instances_remain(self):
        """
        Initialize the self._instances_remain dictionary
        :return: None
        """
        self._instances_ramain = copy.deepcopy(self._instances.get_dataset())

        # b = self._instances.get_dataset()
        #
        # assert id(self._instances_ramain) != id(b)
        #
        # exit(1)


    def _select_inst_wrk(self):
        """
        Select the next instance to label and the corresponding next worker
        :return: the selected instance and the worker
        """
        prior_params_ab = self._param_of_all_insts
        prior_params_cd = self._param_of_all_wrks
        # initialize the R_max and the dic whose form is {R_max:[instance id, worker id]}
        R_max = None
        inst_wrk = []
        # traversing all the available instances and the corresponding workers who have donated labelings.

        assert len(self._instances_ramain) > 0
        for key_ in self._instances_ramain.keys():
            wrks_list = self._instances_ramain[key_]['workers']
            assert len(wrks_list) > 0
            for wrk_pos, wrk in enumerate(wrks_list):
                [c, d] = prior_params_cd[wrk]
                [a, b] = prior_params_ab[key_]
                # calculate a_tilde and b_tilde
                new_a, new_b = new_a_b(a, b, c, d, 1)
                # calculate I(a,b) and I(a_tilde, b_tilde)
                I_ab = Beta_ab_cdf(a, b)
                I_ab_new = Beta_ab_cdf(new_a, new_b)
                # calculate R1 when z == 1
                R1 = h_function(I_ab_new) - h_function(I_ab)

                # similarly, calculate R2 when z == 0
                new_a, new_b = new_a_b(a, b, c, d, 0)
                I_ab = Beta_ab_cdf(a, b)
                I_ab_new = Beta_ab_cdf(new_a, new_b)
                R2 = h_function(I_ab_new) - h_function(I_ab)

                if R1 >= R2:
                    R = R1
                else:
                    R = R2
                # save the max reward and the instance id, worker id.
                if R_max == None:
                    R_max = R
                    inst_wrk = [key_, wrk, wrk_pos]
                else:
                    if R > R_max:
                        R_max = R
                        inst_wrk = [key_, wrk, wrk_pos] #here I first use a temporary dictionary to save the best R_max:[orig_id, wrk_id],it's wrong

        return R_max, inst_wrk


    def _acquire_label_Update_posterior(self, R_max, inst_wrk):
        """
        After we decide the next instance and the worker, we aquire the label z of the i-th instance from the j-th worker
        and update the corresponding a, b, c, d.
        :return: None
        """
        # Acquire the label
        try:
            [task_id, wrk_id, wrk_pos] = inst_wrk
        except:
            print()
        response_idx = self._instances.get_dataset()[task_id]['workers'].index(wrk_id)
        z_real = self._instances.get_dataset()[task_id]['response'][response_idx]
        [a, b] = self._param_of_all_insts[task_id]
        [c, d] = self._param_of_all_wrks[wrk_id]
        self._instances.update_parameter_a_b(task_id, a, b, c, d, z_real)
        self._workers.update_parameter_c_d(wrk_id, a, b, c, d, z_real)

        return task_id, wrk_pos


    def _pop_out_chosen_wrk_response(self,task_id, idx):
        """
        :parameter: task_id, the selected instance at stage T
                    idx: the worker's index in the list
        Pop the chosen worker and his response from the instances_remain dictionary
        :return: None
        """
        # delete the chosen worker and his response from the instance_remain set
        self._instances_ramain[task_id]['workers'].pop(idx)
        self._instances_ramain[task_id]['response'].pop(idx)
        # add the task_id to self._chosen_inst
        self._chosen_inst.append(task_id)
        # judge whether all the workers of instance task_id have been chosen
        # if yes, pop out the instance from the instances_remain dataset
        if len(self._instances_ramain[task_id]['workers']) == 0:
            self._instances_ramain.pop(task_id)


    def _output_set_Ht(self):
        """
        The Budget T has run out and we output the positive set H_T,the content of
        the Ht is the orig_id of the instances whose a > b
        :return: Ht, Ht_complement
        """
        all_prior_params_ab = self._param_of_all_insts
        chosen_inst_unique = list(set(self._chosen_inst))
        Ht = []
        Ht_complement = []
        for key_ in chosen_inst_unique:
            [a, b] = all_prior_params_ab[key_]
            if a >= b:
                Ht.append(key_)
            else:
                Ht_complement.append(key_)

        return Ht, Ht_complement


    def run_Opt_KG(self):
        """
        Whole algorithm Opt-KG
        :return: Positive set H_T
        """
        Budget_T = self._T
        self._initialize_instances_remain()
        for t in range(0, Budget_T):
            R_max, inst_wrk = self._select_inst_wrk()
            task_id, idx = self._acquire_label_Update_posterior(R_max, inst_wrk)
            self._pop_out_chosen_wrk_response(task_id, idx)

        H_T = self._output_set_Ht()

        return H_T

# test the Algorithm.py
def _test_algorithm():
    filename = 'rte.standardized.tsv'
    a0 = 1
    b0 = 1
    datasource = DataSource(filename, a0, b0)
    workers = Worker(filename, 4, 1)
    Budget = 6400
    Opt_KG = Algorithm(datasource, workers, Budget)
    H_T = Opt_KG.run_Opt_KG()
    print(H_T)


if __name__ == "__main__":
    _test_algorithm()