from hmmPackage.hmm import Hmm
from hmmPackage.extmath import ExtMath
import unittest


class HmmTest(unittest.TestCase):

    def testHmm(self):
        o_hmm = Hmm()
        states = ("Rain", "NoRain")
        possible_observation = ("Umbrella", "NoUmbrella")
        observations = ("Umbrella", "Umbrella",
                        "NoUmbrella", "Umbrella", "Umbrella")
        o_hmm.set_model(states, possible_observation, observations)
        print(o_hmm.__repr__())
        o_hmm.set_init_prob("Rain", 0.5)
        o_hmm.set_init_prob("NoRain", 0.5)

        o_hmm.set_trs_prob(["Rain", "Rain"], 0.7)
        o_hmm.set_trs_prob(["Rain", "NoRain"], 0.3)
        o_hmm.set_trs_prob(["NoRain", "Rain"], 0.3)
        o_hmm.set_trs_prob(["NoRain", "NoRain"], 0.7)

        o_hmm.set_ems_prob(["Rain", "Umbrella"], 0.9)
        o_hmm.set_ems_prob(["Rain", "NoUmbrella"], 0.1)
        o_hmm.set_ems_prob(["NoRain", "Umbrella"], 0.2)
        o_hmm.set_ems_prob(["NoRain", "NoUmbrella"], 0.8)

        print(o_hmm.__repr__())

        o_hmm.fwd()
        o_hmm.bkw()
        o_hmm.fwd_bkw()
        o_hmm.baum_welch()

        DELTA = 1e-4
        # test forward probability
        # https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        self.assertAlmostEqual(0.8182, ExtMath.eexp(
            o_hmm.lnp_alpha[0][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.1818, ExtMath.eexp(
            o_hmm.lnp_alpha[0][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.8834, ExtMath.eexp(
            o_hmm.lnp_alpha[1][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.1166, ExtMath.eexp(
            o_hmm.lnp_alpha[1][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.1907, ExtMath.eexp(
            o_hmm.lnp_alpha[2][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.8093, ExtMath.eexp(
            o_hmm.lnp_alpha[2][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.7308, ExtMath.eexp(
            o_hmm.lnp_alpha[3][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.2692, ExtMath.eexp(
            o_hmm.lnp_alpha[3][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.8673, ExtMath.eexp(
            o_hmm.lnp_alpha[4][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.1327, ExtMath.eexp(
            o_hmm.lnp_alpha[4][states[1]]), delta=DELTA)

        # test forward probability
        # https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
        self.assertAlmostEqual(0.6273, ExtMath.eexp(
            o_hmm.lnp_beta[4][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.3727, ExtMath.eexp(
            o_hmm.lnp_beta[4][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.6533, ExtMath.eexp(
            o_hmm.lnp_beta[3][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.3467, ExtMath.eexp(
            o_hmm.lnp_beta[3][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.3763, ExtMath.eexp(
            o_hmm.lnp_beta[2][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.6237, ExtMath.eexp(
            o_hmm.lnp_beta[2][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.5923, ExtMath.eexp(
            o_hmm.lnp_beta[1][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.4077, ExtMath.eexp(
            o_hmm.lnp_beta[1][states[1]]), delta=DELTA)
        self.assertAlmostEqual(0.6469, ExtMath.eexp(
            o_hmm.lnp_beta[0][states[0]]), delta=DELTA)
        self.assertAlmostEqual(0.3531, ExtMath.eexp(
            o_hmm.lnp_beta[0][states[1]]), delta=DELTA)

        # # test forward-backward probability
        # https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
        # self.assertEquals(0.6469, ExtMath.eexp(
        #     o_hmm.lnp_gamma[0][states[0]]), DELTA)
        # self.assertEquals(0.3531, ExtMath.eexp(
        #     o_hmm.lnp_gamma[0][states[1]]), DELTA)
        # self.assertEquals(0.8673, ExtMath.eexp(
        #     o_hmm.lnp_gamma[1][states[0]]), DELTA)
        # self.assertEquals(0.1327, ExtMath.eexp(
        #     o_hmm.lnp_gamma[1][states[1]]), DELTA)
        # self.assertEquals(0.8204, ExtMath.eexp(
        #     o_hmm.lnp_gamma[2][states[0]]), DELTA)
        # self.assertEquals(0.1796, ExtMath.eexp(
        #     o_hmm.lnp_gamma[2][states[1]]), DELTA)
        # self.assertEquals(0.3075, ExtMath.eexp(
        #     o_hmm.lnp_gamma[3][states[0]]), DELTA)
        # self.assertEquals(0.6925, ExtMath.eexp(
        #     o_hmm.lnp_gamma[3][states[1]]), DELTA)
        # self.assertEquals(0.8204, ExtMath.eexp(
        #     o_hmm.lnp_gamma[4][states[0]]), DELTA)
        # self.assertEquals(0.1796, ExtMath.eexp(
        #     o_hmm.lnp_gamma[4][states[1]]), DELTA)


if __name__ == '__main__':
    unittest.main()
