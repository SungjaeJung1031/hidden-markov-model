"""[summary]

    Returns:
        [type]: [description]
    """


class Hmm:

    def __init__(self):
        """Constructor method
        """
        self.i_sz_sequence = int(0)
        self.i_sz_hidden = int(0)
        self.i_sz_observation = int(0)
        self.f_smoothing_factor = float(0.0)
        self.fd_initial_probability = {}
        self.fd_transition_probability = {}
        self.fd_emission_probability = {}
        self.fd_alpha_probability = {}
        self.fd_beta_probability = {}

    def __repr__(self):
        """Returns the overview of the HMM
        returns: object representation message
        """
        model_overview = ""
        model_overview += "====================================================\n"
        model_overview += "================== Model Overview ==================\n"
        model_overview += "====================================================\n"
        model_overview += "# of sequence: {}\n".format(self.i_sz_sequence)
        model_overview += "# of hidden states: {}\n".format(self.i_sz_hidden)
        model_overview += "# of observations: {}\n".format(
            self.i_sz_observation)
        model_overview += "# smoothing factor: {}\n".format(
            self.f_smoothing_factor)
        model_overview += "----------------------------------------------------\n"
        model_overview += "---------------- Initial Probability ---------------\n"
        model_overview += "----------------------------------------------------\n"
        if self.i_sz_hidden > 0:
            for i in range(self.i_sz_hidden):
                model_overview += "s[{}]: {}\n".format(i,
                                                       self.fd_initial_probability[i])
        else:
            print("NOTICE: initial probability has not been set")

        model_overview += "----------------------------------------------------\n"
        model_overview += "-------------- Transition Probability --------------\n"
        model_overview += "----------------------------------------------------\n"
        if self.i_sz_hidden > 0:
            for i in range(self.i_sz_hidden):
                for j in range(self.i_sz_hidden):
                    model_overview += "(s[{}]->s[{}]: {})  ".format(
                        i, j, self.fd_transition_probability[i][j])
                model_overview += "\n"
        else:
            print("NOTICE: transition probability has not been set")

        model_overview += "----------------------------------------------------\n"
        model_overview += "--------------- Emission Probability ---------------\n"
        model_overview += "----------------------------------------------------\n"
        if self.i_sz_hidden > 0 and self.i_sz_observation > 0:
            for i in range(self.i_sz_observation):
                for j in range(self.i_sz_hidden):
                    model_overview += "(o[{}]->s[{}]: {})  ".format(
                        i, j, self.fd_emission_probability[i][j])
                model_overview += "\n"
        else:
            print("NOTICE: emission probability has not been set")
        model_overview += "====================================================\n"

        return model_overview

    def set_model(self, i_sz_seq, i_sz_hidden, i_sz_obs, f_smoothing_factor=0.0):
        """set HMM parameters

        Args:
            i_sz_seq (int): the size of the (observation) sequences.
            i_sz_hidden (int): the size of the hidden states.
            i_sz_obs (int): the size of the observations.
            f_smoothing_factor (float, optional): smoothing factor of probabilities(alpha or beta probability). Defaults to 0.0.
        """

        self.i_sz_sqeunce = int(i_sz_seq)
        self.i_sz_hidden = int(i_sz_hidden)
        self.i_sz_observation = int(i_sz_obs)
        self.f_smoothing_factor = float(f_smoothing_factor)
        self.fd_initial_probability = {i: 0.0 for i in range(self.i_sz_hidden)}
        self.fd_transition_probability = {
            i: {j: 0.0 for j in range(self.i_sz_hidden)} for i in range(self.i_sz_hidden)}
        self.fd_emission_probability = {
            i: {j: 0.0 for j in range(self.i_sz_hidden)} for i in range(self.i_sz_observation)}
        self.fd_alpha_probability = {
            i: {j: 0.0 for j in range(self.i_sz_hidden)} for i in range(self.i_sz_sequence)
        }

    def forward_algorithm(self, l_obs):
        """forward algorithm

        Args:
            l_obs (list): list of observation
        """

        for i_idx_hidden in range(self.i_sz_hidden):
            self.fd_alpha_probability[0][i_idx_hidden] \
                = self.fd_initial_probability[i_idx_hidden] \
                * self.fd_emission_probability[l_obs[0]][i_idx_hidden]

        for i_idx_seq in range(1, self.i_sz_sequence):
            for i_idx_hidden_i in range(self.i_sz_hidden):
                self.fd_alpha_probability[i_idx_seq][i_idx_hidden] = 0.0

                for i_idx_hidden_j in range(self.i_sz_hidden):
                    self.fd_alpha_probability[i_idx_seq][i_idx_hidden] \
                        += self.fd_alpha_probability[i_idx_seq-1][i_idx_hidden_j] \
                        * self.fd_transition_probability[i_idx_hidden_j][i_idx_hidden_i]

                self.fd_alpha_probability \
                    *= self.fd_emission_probability[l_obs[i_idx_seq]][i_idx_hidden]

    def bacward_algorithm(self):
        """backward algorithm
        """
        pass

    def UpdateParameters(self):
        """update parameter
        """
        pass
