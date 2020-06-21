""" Hidden Markov Model
    """
import math
from .extmath import ExtMath


class Hmm:

    def __init__(self):
        """Constructor method
        """
        self.states = None
        self.possible_observations = None
        self.observations = None
        self.smoothing_factor = float(0.0)
        self.p_init = {}
        self.p_trans = {}
        self.p_ems = {}
        self.lnp_alpha = {}
        self.lnp_beta = {}
        self.lnp_gamma = {}
        self.lnp_xi = {}

    def __repr__(self):
        """Returns the overview of the HMM

        Returns:
            str: object representation
        """

        model_overview = ""
        model_overview += "====================================================\n"
        model_overview += "================== Model Overview ==================\n"
        model_overview += "====================================================\n"
        model_overview += "# smoothing factor: {}\n".format(
            self.smoothing_factor)

        model_overview += "----------------------------------------------------\n"
        model_overview += "----------------------- States ---------------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0:
            for i, state_i in enumerate(self.states):
                model_overview += "state #{}: {}\n".format(
                    i, state_i)
        else:
            model_overview += "NOTICE: the size of the hidden states is 0"

        model_overview += "----------------------------------------------------\n"
        model_overview += "--------------- Possible Observations --------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0:
            for i, possbile_observation_i in enumerate(self.possible_observations):
                model_overview += "possible observation #{}: {}\n".format(
                    i, possbile_observation_i)
        else:
            model_overview += "NOTICE: the size of the hidden states is 0"

        model_overview += "----------------------------------------------------\n"
        model_overview += "---------------- Initial Probability ---------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0:
            for i, (init_prob_k, init_prob_v) in enumerate(self.p_init.items()):
                model_overview += "s[{}]: {}\n".format(init_prob_k,
                                                       init_prob_v)
        else:
            model_overview += "NOTICE: initial probability has not been set\n"

        model_overview += "----------------------------------------------------\n"
        model_overview += "-------------- Transition Probability --------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0:
            for i, (trans_prob_k, trans_prob_v) in enumerate(self.p_trans.items()):
                for j, (trans_prob_i_k, trans_prob_i_v) in enumerate(trans_prob_v.items()):
                    model_overview += "(s[{}]->s[{}]: {})  ".format(
                        trans_prob_k, trans_prob_i_k, trans_prob_i_v)
                model_overview += "\n"
        else:
            model_overview += "NOTICE: transition probability has not been set\n"

        model_overview += "----------------------------------------------------\n"
        model_overview += "--------------- Emission Probability ---------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0 and len(self.possible_observations) > 0:
            for i, (ems_prob_k, ems_prob_v) in enumerate(self.p_ems.items()):
                for j, (ems_prob_i_k, ems_prob_i_v) in enumerate(ems_prob_v.items()):
                    model_overview += "(s[{}]->o[{}]: {})  ".format(
                        ems_prob_k, ems_prob_i_k, ems_prob_i_v)
                model_overview += "\n"
        else:
            model_overview += "NOTICE: emission probability has not been set"
        model_overview += "====================================================\n"

        return model_overview

    def set_model(self, states, possible_observations, observations, smoothing_factor=0.0):
        """set HMM parameters

        Args:
            sz_seq (int): the size of the (observation) sequences.
            sz_hidden (int): the size of the hidden states.
            sz_obs (int): the size of the observations.
            smoothing_factor (float, optional): smoothing factor of probabilities(alpha or beta probability). Defaults to 0.0.
        """
        self.states = states
        self.possible_observations = possible_observations
        self.observations = observations
        self.smoothing_factor = float(smoothing_factor)
        self.p_init = {state: 0.0 for state in self.states}
        self.p_trans = {
            state_i: {state_j: 0.0 for state_j in self.states} for state_i in self.states}
        self.p_ems = {
            state: {possible_observation: 0.0 for possible_observation in self.possible_observations} for state in self.states}
        self.lnp_alpha = {
            obs: {state: 0.0 for state in self.states} for obs in range(len(self.observations))}
        self.lnp_beta = {
            obs: {state: 0.0 for state in self.states} for obs in range(len(self.observations))}
        self.lnp_gamma = {
            idx_obs: {state: 0.0 for state in self.states} for idx_obs in range(len(self.observations))}
        self.lnp_xi = {
            idx_obs: {state_i: {state_j: 0.0 for state_j in self.states} for state_i in self.states} for idx_obs in range(len(self.observations))
        }

    def baum_welch(self):
        """Baum-Welch algorithm
        """
        for idx_obs in range(len(self.observations)):
            if idx_obs == (len(self.observations) - 2):
                break
            normalizer = math.nan
            for state_i in self.states:
                for state_j in self.states:
                    idx_next_obs = idx_obs+1
                    first_factor = ExtMath.eln(self.p_trans[state_i][state_j])

                    second_factor = ExtMath.eln_product(
                        self.p_ems[state_j][self.observations[idx_next_obs]],
                        self.lnp_beta[idx_next_obs][state_j])

                    self.lnp_xi[idx_obs][state_i][state_j] = ExtMath.eln_product(
                        self.lnp_alpha[idx_obs][state_i], ExtMath.eln_product(first_factor, second_factor))

                    normalizer = ExtMath.eln_sum(
                        normalizer, self.lnp_xi[idx_obs][state_i][state_j])

            for state_i in self.states:
                for state_j in self.states:
                    self.lnp_xi[idx_obs][state_i][state_j] = ExtMath.eln_product(
                        self.lnp_xi[idx_obs][state_i][state_j], -normalizer)

    def fwd_bkw(self):
        """numerically stable forward-backward algorithm
        """
        for idx_obs in range(len(self.observations)):
            normalizer = math.nan
            for state in self.states:
                self.lnp_gamma[idx_obs][state] = ExtMath.eln_product(
                    self.lnp_alpha[idx_obs][state], self.lnp_beta[idx_obs][state])
                normalizer = ExtMath.eln_sum(
                    normalizer, self.lnp_gamma[idx_obs][state])

            for state in self.states:
                self.lnp_gamma[idx_obs][state] = ExtMath.eln_product(
                    self.lnp_gamma[idx_obs][state], -normalizer)

    def fwd(self):
        """numerically stable forward algorithm
        """
        for idx_obs, obs_i in enumerate(self.observations):
            ln_normalizer = math.nan
            for state_j in self.states:
                log_alpha = math.nan
                if idx_obs == 0:
                    log_alpha = ExtMath.eln(self.p_init[state_j])
                else:
                    idx_prev_obs = idx_obs - 1
                    for state_i in self.states:
                        first_factor = self.lnp_alpha[idx_prev_obs][state_i]
                        second_factor = ExtMath.eln(
                            self.p_trans[state_i][state_j])
                        log_alpha = ExtMath.eln_sum(
                            log_alpha, ExtMath.eln_product(first_factor, second_factor))

                self.lnp_alpha[idx_obs][state_j] = ExtMath.eln_product(
                    log_alpha, ExtMath.eln(self.p_ems[state_j][obs_i]))

                # update normalizer
                ln_normalizer = ExtMath.eln_sum(
                    ln_normalizer, self.lnp_alpha[idx_obs][state_j])

            # normalization
            for state_j in self.states:
                self.lnp_alpha[idx_obs][state_j] = ExtMath.eln_product(
                    self.lnp_alpha[idx_obs][state_j], -ln_normalizer)

    def bkw(self):
        """numerically stable backward algorithm
        """
        for idx_obs in reversed(range(len(self.observations))):
            ln_normalizer = math.nan
            for state_i in self.states:
                ln_beta = math.nan
                for state_j in self.states:
                    lnp_ems = ExtMath.eln(
                        self.p_ems[state_j][self.observations[idx_obs]])
                    lnp_next_beta = 0.0
                    if idx_obs != len(self.observations)-1:
                        idx_next_obs = idx_obs + 1
                        lnp_next_beta = self.lnp_beta[idx_next_obs][state_j]

                    ln_beta = ExtMath.eln_sum(ln_beta,
                                              ExtMath.eln_product(ExtMath.eln(self.p_trans[state_i][state_j]),
                                                                  ExtMath.eln_product(lnp_ems, lnp_next_beta)))

                self.lnp_beta[idx_obs][state_i] = ln_beta

                # update normalizer
                ln_normalizer = ExtMath.eln_sum(
                    ln_normalizer, self.lnp_beta[idx_obs][state_i])

            # normalization
            for state_i in self.states:
                self.lnp_beta[idx_obs][state_i] = ExtMath.eln_product(
                    self.lnp_beta[idx_obs][state_i], -ln_normalizer)

    def set_init_prob(self, key, val):
        """set initial probability

        Args:
            key (Any): key (index of a hidden state) of the dictionary for the initial probabiliy
            val (float): value (initial probability of a hidden state) of the dictionary for the initial probability
        """
        self.p_init[key] = val

    def set_trs_prob(self, key, val):
        """set transition probability

        Args:
            key (list): key (index of hidden states) of the dictionary for the transition probabilty
                           key[0] is for the index of the previous hidden state
                           key[1] is for the index of the next hidden state
            val (float): value (transition probability) of the dictionary for the initial probability
        """
        self.p_trans[key[0]][key[1]] = val

    def set_ems_prob(self, key, val):
        """set emission probability

        Args:
            key (list): key of the dictionary for the emission probability
                           key[0] is for the index of a observation
                           key[1] is for the index of a hidden state
            val ([type]): value (emission probability) of the dictionary for the transition probability
        """
        self.p_ems[key[0]][key[1]] = val

    def update_parameters(self):
        """update parameter
        """
        pass
