# nAMI
Scripts for mixing acoustic events with AMI corpus MDM recordings.

The repository presents scripts for spatial reverberation of the noise database (simulation.py) and for mixing reverberated noise to the MDM8 data of the AMI corpus (add_noise_ami.py). The algorithms are discussed in paper [1]. For the noise database the standard Freesound database is utilized.

Corpus mixing has several stochastic components, thus one-to-one reproduction of the corpus, the evaluation results of which were presented in paper [1], is impossible. For one-to-one reproduction it is advised to use the logs of mixing, and apply the same augmentation and mixing parameters.

[1] S. Astapov, G. Svirskiy, A. Lavrentyev, T. Prisyach, D. Popov, D. Ubskiy, and V. Kabarov, "Acoustic Event Mixing to Multichannel AMI Data for Distant Speech Recognition and Acoustic Event Classification Benchmarking," In: Proc. Int. Conf. on Speech and Computer (SPECOM 2019), pp. 31-42, Sept. 2019. https://doi.org/10.1007/978-3-030-26061-3_4
