# hmm-lib

## About

This library implements Hidden Markov Models (HMM) for time-inhomogeneous Markov processes. This means that, in contrast to many other HMM implementations, there can be different states and a different transition matrix at each time step.

This library provides an implementation of

The Viterbi algorithm, which computes the most likely sequence of states.
The forward-backward algorithm, which computes the probability of all state candidates given the entire sequence of observations. This process is also called smoothing.

## Setup

TBD

## Style Guide

The documentation of the code follows [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

## License

This library is licensed under the [MIT License](https://opensource.org/licenses/MIT)

## Changes

- 1.0.1:
  - tbd.
- 1.0.0:
  - Below algorithms are updated for the numerical stability
    - forward algorithm
    - backward algorithm
    - forward-backward algorithm
    - Baum-Welch algorithm
  - test file is added for the below algorithms
    - forward algorithm
    - backward algorithm
- 0.0.1
  - Below algorithms are added
    - forward algorithm
    - backward algorithm

## Contribute

Contribute are alway welcome.

## References

- [L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989, doi: 10.1109/5.18626.](https://ieeexplore.ieee.org/document/18626/citations#citations)
- [Tobias P. Mann,"Numerically stable hidden Markov model implementation", Feb, 2006](https://www.semanticscholar.org/paper/Numerically-Stable-Hidden-Markov-Model-Mann/109bb95ffec81185f2c69f654711b25b8348adb0#related-papers)
