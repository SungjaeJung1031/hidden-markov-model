from hmmPackage import hmm
import unittest


class HmmTest(unittest.TestCase):

    def testHmm(self):
        o_hmm = hmm.Hmm()
        states = ("Healthy", "Fever", "E")
        possible_observation = ("Normal", "Cold", "Dizzy")
        observations = ("Normal", "Cold", "Dizzy")
        o_hmm.set_model(states, possible_observation, observations)
        print(o_hmm.__repr__())
        o_hmm.set_init_prob("Healthy", 0.6)
        o_hmm.set_init_prob("Fever", 0.4)

        o_hmm.set_trs_prob(["Healthy", "Healthy"], 0.69)
        o_hmm.set_trs_prob(["Healthy", "Fever"], 0.3)
        o_hmm.set_trs_prob(["Healthy", "E"], 0.01)
        o_hmm.set_trs_prob(["Fever", "Healthy"], 0.4)
        o_hmm.set_trs_prob(["Fever", "Fever"], 0.59)
        o_hmm.set_trs_prob(["Fever", "E"], 0.01)

        o_hmm.set_ems_prob(["Healthy", "Normal"], 0.5)
        o_hmm.set_ems_prob(["Healthy", "Cold"], 0.4)
        o_hmm.set_ems_prob(["Healthy", "Dizzy"], 0.1)
        o_hmm.set_ems_prob(["Fever", "Normal"], 0.1)
        o_hmm.set_ems_prob(["Fever", "Cold"], 0.3)
        o_hmm.set_ems_prob(["Fever", "Dizzy"], 0.6)

        print(o_hmm.__repr__())


if __name__ == '__main__':
    unittest.main()
