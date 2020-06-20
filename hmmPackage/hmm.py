"""[summary]

    Returns:
        [type]: [description]
    """


class Hmm:

    def __init__(self):
        """Constructor method
        """
        self.states = None
        self.possible_observations = None
        self.observations = None
        self.smoothing_factor = float(0.0)
        self.init_prob = {}
        self.trans_prob = {}
        self.ems_prob = {}
        self.alpha_prob = {}
        self.beta_prob = {}
        self.gamma_prob = {}

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
            for i, (init_prob_k, init_prob_v) in enumerate(self.init_prob.items()):
                model_overview += "s[{}]: {}\n".format(init_prob_k,
                                                       init_prob_v)
        else:
            model_overview += "NOTICE: initial probability has not been set\n"

        model_overview += "----------------------------------------------------\n"
        model_overview += "-------------- Transition Probability --------------\n"
        model_overview += "----------------------------------------------------\n"
        if len(self.states) > 0:
            for i, (trans_prob_k, trans_prob_v) in enumerate(self.trans_prob.items()):
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
            for i, (ems_prob_k, ems_prob_v) in enumerate(self.ems_prob.items()):
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
        self.init_prob = {state: 0.0 for state in self.states}
        self.trans_prob = {
            state_i: {state_j: 0.0 for state_j in self.states} for state_i in self.states}
        self.ems_prob = {
            state: {possible_observation: 0.0 for possible_observation in self.possible_observations} for state in self.states}
        self.alpha_prob = {
            i: {state: 0.0 for state in self.states} for i in len(self.observations)}
        self.beta_prob = {
            i: {state: 0.0 for state in self.states} for i in len(self.observations)}
        self.gamma_prob = {
            i: {state: 0.0 for state in self.states} for i in len(self.observations)}

    def forward_backward_algorithm(self):
        """forward-backward algorithm
        """
        self.forward_algorithm()

        # TODO: calculate sum of alpha prob

        self.backward_algorithm()

        # TODO: calculate sum of beta prob

        # TODO: calculate gamma (posterior) prob

    def forward_algorithm(self):
        """forawrd algorithm
        """
        for idx_obs, obs_i in enumerate(self.observations):
            prev_event_prob = 0.0
            for state_i in self.states:
                if idx_obs == 0:
                    prev_event_prob = self.init_prob[state_i]
                else:
                    idx_prev_obs = idx_obs-1
                    prev_event_prob = sum(
                        self.alpha_prob[idx_prev_obs][state_j]
                        * self.trans_prob[state_j][state_i]
                        for state_j in self.states)

                self.alpha_prob[idx_obs][state_i] = prev_event_prob\
                    * self.ems_prob[state_i][obs_i]

    def backward_algorithm(self):
        """backward algorithm
        """
        for idx_obs in reversed(range(len(self.observations))):
            cur_beta_prob = 0.0
            for state_i in self.states:
                if idx_obs == (len(self.observations)-1):
                    cur_beta_prob = self.trans_prob[state_i]["E"]
                else:
                    idx_next_obs = idx_obs+1
                    cur_beta_prob = sum(
                        self.beta_prob[idx_next_obs][state_j]
                        * self.trans_prob[state_i][state_j]
                        * self.ems_prob[state_j][self.observations[idx_next_obs]]
                        for state_j in self.states
                    )

                self.beta_prob[idx_obs][state_i] = cur_beta_prob

        # def forward_algorithm_old(self, l_obs):
        #     """forward algorithm

        #     Args:
        #         l_obs (list): list of observation
        #     """
        #     fd_norm_factor = {}

        #     # step #2
        #     for idx_seq in range(1, self.sz_sequence):
        #         fd_norm_factor[idx_seq] = 0.0
        #         for idx_hidden_i in range(self.sz_hidden):
        #             self.alpha_prob[idx_seq][idx_hidden] = 0.0

        #             for idx_hidden_j in range(self.sz_hidden):
        #                 self.alpha_prob[idx_seq][idx_hidden] \
        #                     += self.alpha_prob[idx_seq-1][idx_hidden_j] \
        #                     * self.trans_prob[idx_hidden_j][idx_hidden_i]

        #             self.alpha_prob \
        #                 *= self.ems_prob[l_obs[idx_seq]][idx_hidden]

        #         fd_norm_factor[idx_seq] = 1.0/fd_nidx_obsorm_factor[idx_seq]

        #         for idx_hidden in range(self.sz_hidden):
        #             self.alpha_prob[idx_seq][idx_hidden] *= fd_norm_factor[idx_seq]

        # def forward_algorithm_old_init(self, l_obs):
        #     """forward algorithm with initiali probability

        #     Args:
        #         i_obs (int): observation
        #     """
        #     norm_factor = 0.0  # normalization factor

        #     for idx_hidden in range(self.sz_hidden):
        #         self.alpha_prob[0][idx_hidden]\
        #             = self.init_prob[idx_hidden] \
        #             * self.ems_prob[l_obs[0]][idx_hidden]

        #         # update normalization factor
        #         norm_factor += self.alpha_prob[0][idx_hidden]

        #     # normalization
        #     norm_factor = 1.0/norm_factor
        #     for idx_hidden in range(self.sz_hidden):
        #         self.alpha_prob[0][idx_hidden] *= norm_factor

        # def forward_algorithm_old(self, l_obs):
        #     """forward algorithm

        #     Args:
        #         l_obs (list): list of observation
        #     """
        #     fd_norm_factor = {}

        #     # step #2
        #     for idx_seq in range(1, self.sz_sequence):
        #         fd_norm_factor[idx_seq] = 0.0
        #         for idx_hidden_i in range(self.sz_hidden):
        #             self.alpha_prob[idx_seq][idx_hidden] = 0.0

        #             for idx_hidden_j in range(self.sz_hidden):
        #                 self.alpha_prob[idx_seq][idx_hidden] \
        #                     += self.alpha_prob[idx_seq-1][idx_hidden_j] \
        #                     * self.trans_prob[idx_hidden_j][idx_hidden_i]

        #             self.alpha_prob \
        #                 *= self.ems_prob[l_obs[idx_seq]][idx_hidden]

        #         fd_norm_factor[idx_seq] = 1.0/fd_norm_factor[idx_seq]

        #         for idx_hidden in range(self.sz_hidden):
        #             self.alpha_prob[idx_seq][idx_hidden] *= fd_norm_factor[idx_seq]

    def set_init_prob(self, key, val):
        """set initial probability

        Args:
            key (Any): key (index of a hidden state) of the dictionary for the initial probabiliy
            val (float): value (initial probability of a hidden state) of the dictionary for the initial probability
        """
        self.init_prob[key] = val

    def set_trs_prob(self, key, val):
        """set transition probability

        Args:
            key (list): key (index of hidden states) of the dictionary for the transition probabilty
                           key[0] is for the index of the previous hidden state
                           key[1] is for the index of the next hidden state
            val (float): value (transition probability) of the dictionary for the initial probability
        """
        self.trans_prob[key[0]][key[1]] = val

    def set_ems_prob(self, key, val):
        """set emission probability

        Args:
            key (list): key of the dictionary for the emission probability
                           key[0] is for the index of a observation
                           key[1] is for the index of a hidden state
            val ([type]): value (emission probability) of the dictionary for the transition probability
        """
        self.ems_prob[key[0]][key[1]] = val

    def bacward_algorithm(self):
        """backward algems_proborithm
        """
        pass

    def update_parameters(self):
        """update parameter
        """
        # for i in range(self.sz_hidden):
        #     for j in range(self.sz_hidden):
        #         self.init_prob = (self.smoothing_factor + )
        pass
