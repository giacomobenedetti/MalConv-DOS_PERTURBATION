# MalConv-DOS_PERTURBATION
Python implementation of the misclassification attack on MalConv Neural Network, it is based on <a href="https://arxiv.org/abs/1901.03583">Explaining Vulnerabilities of Deep Learning to Adversarial Malware Binaries</a>, proposed by <a href="https://csec.it/people/luca_demetrio/">Demetrio</a> et al.

# Usage
It simply requires a binary sample as argument and it will return the DOS header before and after the attack, so: <br>
`python attack.py <binary>` <br>
At the end of interations a graph of model accuracy during iterations will be provided.
