# File: ccn_model.py
# Author: Christopher A. Wood

import sys
from math import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True
import time
import numpy as np
from random import *

# References:
# [1] - Modeling data transfer in content-centric networking (extended version)

# Model parameters used to specify constraints on the topology/scenario being modeled
# They're populated down at the bottom
rts = 0
N = 0
M = 0
K = 0
c = 0
x = 0
alpha = 0
delta = 0
sigma = 0
sigma_k = []
lmbda = 0
lmbda_k = []
pop_dist = "zipf"

### MODEL EQUATIONS ###

def calc_g():
	''' Calculate g according to equation (2) in [1]

		Status: Correct.
	'''
	global lmbda, alpha, sigma, M, K, c

	m = float(M) / float(K)
	g = lmbda * c * pow(sigma, alpha) * pow(m, (alpha - 1.0)) * pow(gamma(1.0 - (1.0 - alpha)), alpha)
	return 1.0 / g

def calc_q(k):
	global c, alpha, pop_dist, K
	if (pop_dist == "zipf"):
		return c / (k ** alpha)
	elif (pop_dist == "uniform"):
		return c / K

def calc_p1(k):
	''' Calculate p_k(1) according to first part of proof of proposition 6.2 in [1]

	 	p_k(1) = e^( (-lambda / m) * q_k * g * x^alpha )

		Status: Correct.
	'''
	global lmbda, alpha, M, K, c, x

	m = float(M) / float(K)
	q_k = calc_q(k)
	g = calc_g()
	p_k1 = exp( ( (lmbda * -1.0) / m ) * q_k * pow(x, alpha) * g )
	return p_k1

def calc_p_filter(k, i, filtering = True):
	''' Calculate p_filt(k,i) according to equation (28) in [1]

		Status: Correct.
	'''
	global lmbda, alpha, sigma, sigma_k, delta, M, K, c, x

	if filtering: # for k \in {1,..,N}
		bki = calc_bk(k, i, filtering)
		# bki_sigma = pow(bki, sigma_k[k - 1])

		# technical report version
		# num = bki * ( 1.0 - bki_sigma )

		# paper version
		num = 1.0 - bki

		# technical report version
		# denom = (1.0 - bki) * sigma_k[k - 1]

		# paper version
		denom = 1.0 - ((1.0 - (1.0 / sigma)) * bki)

		frac = num / denom

		# tecnical report
		# return (1.0 - frac)

		# paper
		return frac
	else:
		raise Exception("Non-filtering model not implemented.")

def calc_bk(k, i, filtering = True):
	''' Calculate b_k(i) according to equation 28 in [1]

		Status: Match.
	'''
	global lmbda, alpha, sigma, sigma_k, delta, M, K, c, x

	if filtering:
		m = float(M / K)
		if (i == 1):
			return exp( (calc_delta(k, i, filtering) * -1) * lmbda_k[k - 1] / m )
		elif i > 1:
			mu_k = calc_mu(k, i - 1, filtering)
			return exp( 2.0 * (calc_delta(k, i, filtering) * -1) * mu_k / m)
	else:
		raise Exception("Non-filtering model not implemented.")

def calc_mu(k, i, filtering = True):
	''' Calculate mu_k(i) according to equation 28 in [1]

		Status: Match.
	'''	
	global lmbda, lmbda_k, alpha, sigma, sigma_k, delta, delta_k

	if filtering:
		if (i == 1):
			return (lmbda_k[k - 1] * calc_p1(k) * (1.0 - calc_p_filter(k, 1, filtering)))
		elif (i > 1):
			return (2.0 * calc_mu(k, i - 1, filtering) * calc_p(k, i, filtering) * (1.0 - calc_p_filter(k, i, filtering)))
		else:
			raise Exception("Invalid router index: " + str(i))
	else:
		raise Exception("Non-filtering model not implemented.")

def calc_p(k, i, filtering = True): # assume filtering (interest aggregation) is enabled
	''' Calculate p_k(i) according to proposition 6.5 in [1].

		p_k(1) = e^([-\lambda / m)] * q_k * g * x^alpha)

		Status: 
	'''
	global lmbda, alpha, sigma, M, K, N, c, x
	
	# M = number of different content items, K = # of content classes, so there's m things in each class k
	m = float(M) / float(K)

	if filtering:
		pk1 = calc_p1(k) # p_k(1)^f \equiv p_k(1) (no previous filtering has occurred)

		# Sanity check
		if (pk1 > 1.0):
			raise Exception("Pk(1) can't be larger than 1.0")

		# Calculate the exponent
		exponent = float(1.0) 
		for l in range(1, i): #[1, i - 1]
			p_k_l = calc_p(k, l, filtering)

			# print(str(i) + " - calculating filter for l = " + str(l))
			p_filter_k_l = calc_p_filter(k, l, filtering)

			# More sanity checks
			if (p_k_l > 1.0):
				raise Exception("ERROR: miss probability invalid: " + str(p_k_l))
			if (p_filter_k_l > 1.0):
				raise Exception("ERROR: filter miss invalid: " + str(p_filter_k_l))

			# Accumulate the exponent
			# print("Round: " + str(l))
			# print(p_k_l)
			# print(p_filter_k_l)
			# print("pre exp: " + str(exponent))

			# FROM TECHNICAL REPORT
			exponent = exponent * (p_k_l * (1.0 - p_filter_k_l))

			# FROM PAPER #1
			# exponent = exponent * p_k_l 

			# # FROM PAPER #2
			# exponent = exponent * (1.0 - p_filter_k_l)

			# print("post exp: " + str(exponent))

		# p = p_k(1) ^ exponent
		return pow(pk1, exponent)
	else:
		pk1 = calc_p1(k) # p_k(1)^f \equiv p_k(1) (no previous filtering has occurred)

		# Sanity check
		if (pk1 > 1.0):
			raise Exception("Pk(1) can't be larger than 1.0")

		# Calculate the exponent
		exponent = float(1.0) 
		for l in range(1, i): #[1, i - 1]
			p_k_l = calc_p(k, l, filtering)

			# More sanity checks
			if (p_k_l > 1.0):
				raise Exception("ERROR: miss probability invalid: " + str(p_k_l))

			# Accumulate the exponent
			exponent = exponent * p_k_l 

		# p = p_k(1) ^ exponent
		return pow(pk1, exponent)

def calc_rt(i): # just assume a link rate of 1ms (global rts can have some pull on this)
	# global rts
	return 2.0 * i 

# the model assumes that all links have the same round trip delay, which is fine
# this is the virtual RTT for a single chunk, which is ultimately part of a piece of content somehow, depending on fragmentation

# Status: Validated.
def calc_VRTT(k, filtering = True):
	global N
	s = 0.0

	# Compute the weighted sum
	print("Calculating VRTT for k = " + str(k) + ": "),
	for i in range(1, N + 1): #[1, N]

		# FIX: if we're at the last node, then we must have hit the producer, and so the content will be there...
		pp = calc_p(k, i, filtering)
		if (i == N):
			pp = 0.0 # we won't miss here.
		if (pp > 1.0):
			raise Exception("Probabilities (p(k, i)) can't be larger than one")

		# Round trip time between consumer and node level i in the tree
		rti = calc_rt(i) # $R_i$

		prod = 1.0
		for j in range(1, i): #[1, i - 1]
			pkj = calc_p(k, j, filtering)
			if (pkj > 1.0):
				raise Exception("Probabilities (pkj) can't be larger than one")
			prod = prod * pkj
		# print("n = " + str(i))
		# print("prod = " + str(prod))
		scale = (rti * (1.0 - pp) * prod)
		# scale = (rti * (pp) * prod) #* (1.0 - prod))
		# print(scale)
		# print("before: " + str(s))
		s = s + scale
		# print("after: " + str(s))
		# s = s + (rti * (1.2 - pp))

	# Safety check
	if (s < 0):
		# print("k = " + str(k))
		raise Exception("Can't have negative VRTT")
	# print(s)

	return s

def calc_RVRTT(k, i, filtering):
	global N
	s = 0.0

	for j in range(i, N + 1): # [i, N]

		if (j == 1):
			filtering = True
		else:
			filtering = False

		rii = (calc_rt(j) - calc_rt(i - 1))
		pp = calc_p(k, j, filtering)
		prod = 1.0
		for l in range(i, j): # [i, j - 1]
			prod = prod * calc_p(k, l, filtering)
		s = s + (rii * (1.0 - pp) * prod)

	return s

def calc_delta(k, i, filtering):
	global delta
	return min(delta, calc_RVRTT(k, i, filtering))

def init_1():
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta

	M = 2000
	K = 1000
	alpha = 2.0
	lmbda = 40.0
	W = 1.0
	sigma = 10.0 * (10**3)   # 10kB chunk size for everything
	x = 2000 * sigma
	c = 1.0 
	N = 3 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 1.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details

def init_2():
	''' This case has perfectly matching miss probabilities and miss rates - but the VRTT still converges to zero - where is the error?
	'''
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	M = 20000
	K = 10000
	alpha = 2.0
	lmbda = 40.0
	W = 1.0
	sigma = 10.0 * (10**3)   # 10kB chunk size for everything
	x = 2000 * sigma
	c = 1.0 
	N = 3 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 1.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details

def init_3():
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	M = 200
	K = 200
	alpha = 2.0
	lmbda = 40.0
	W = 1.0
	sigma = 10   # 10kB chunk size for everything
	x = 200 * sigma
	c = 1.0 
	N = 5 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 1.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details

def init_4():
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	M = 200
	K = 200
	alpha = 2.0
	lmbda = 40.0
	W = 1.0
	sigma = 10   # 10kB chunk size for everything
	x = 2 * sigma
	c = 1.0 
	N = 5 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 1.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details

def init_5_1():
	''' Copy of 2, but trying to get the expected time in better shape...
	'''
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	# M = 20000
	# K = 10000
	M = 20000
	K = 400
	alpha = 1.6
	lmbda = 40.0
	W = 1.0
	sigma = 10.0       # 10kB chunk size for everything
	x = 200 * sigma
	c = 1
	N = 5 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 10.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details
		# lmbda_k.append(expovariate(lmbda * calc_q(i + 1)))
		# lmbda_k.append(np.random.poisson(lmbda * calc_q(i + 1), 1))

	# Do the calculations
	n1 = []
	mu1 = []
	vrtt1 = []
	n2 = []
	mu2 = []
	vrtt2 = []
	n3 = []
	mu3 = []
	vrtt3 = []
	vrtt = []
	n4 = []
	n5 = []
	mu4 = []
	mu5 = []
	for k in range(1, K + 1):
		vrtt.append(calc_VRTT(k))

		# miss rates
		mu1.append(calc_mu(k, 1, True) * sigma)
		mu2.append(calc_mu(k, 2, True) * sigma)
		mu3.append(calc_mu(k, 3, True) * sigma)
		mu4.append(calc_mu(k, 4, True) * sigma)
		mu5.append(calc_mu(k, 5, True) * sigma)
		
		# miss probabilities
		n1.append(calc_p(k, 1))
		n2.append(calc_p(k, 2))
		n3.append(calc_p(k, 3))
		n4.append(calc_p(k, 4))
		n5.append(calc_p(k, 5))

	# Show the plot
	x = range(K)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# VRTT
	ax1.scatter(x, vrtt, s=10, c='k', marker="o")
	plt.xlabel(r'Popularity Class $k$')
	plt.ylabel(r'VRTT')
	plt.title(r'\textbf{VRTT Characterization')

	# Miss rates
	# ax1.scatter(x, mu1, s=10, c='g', marker="o", label=r'$\mu_1$')
	# ax1.scatter(x, mu2, s=10, c='r', marker="o", label=r'$\mu_2$')
	# ax1.scatter(x, mu3, s=10, c='b', marker="o", label=r'$\mu_3$')
	# ax1.scatter(x, mu4, s=10, c='m', marker="o", label=r'$\mu_4$')
	# ax1.scatter(x, mu5, s=10, c='k', marker="o", label=r'$\mu_5$')
	# ax1.legend(loc=0, scatterpoints = 5)
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'Miss Rate $\mu$ (chunks/s)')
	# plt.title(r'\textbf{Miss Rate Characterization')

	# Miss probabilities
	# ax1.scatter(x, n1, s=10, c='g', marker="o", label=r'$p_k(1)$')
	# ax1.scatter(x, n2, s=10, c='r', marker="o", label=r'$p_k(2)$')
	# ax1.scatter(x, n3, s=10, c='b', marker="o", label=r'$p_k(3)$')
	# ax1.scatter(x, n4, s=10, c='m', marker="o", label=r'$p_k(4)$')
	# ax1.scatter(x, n5, s=10, c='k', marker="o", label=r'$p_k(5)$')
	# ax1.legend(loc=0, scatterpoints = 5)
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'Miss Probability $p_k(i)$')
	# plt.title(r'\textbf{Miss Probability Characterization')

	# Render the figure for display...
	plt.show()

def init_5_1_uniform():
	''' Copy of 2, but trying to get the expected time in better shape...
	'''
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta, pop_dist

	# M = 20000
	# K = 10000
	pop_dist = "uniform"
	M = 20000
	K = 400
	alpha = 1.6
	lmbda = 40.0
	W = 1.0
	sigma = 10.0       # 10kB chunk size for everything
	x = 200 * sigma
	c = 1
	N = 5 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 10.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details
		# lmbda_k.append(expovariate(lmbda * calc_q(i + 1)))
		# lmbda_k.append(np.random.poisson(lmbda * calc_q(i + 1), 1))

	# Do the calculations
	n1 = []
	mu1 = []
	vrtt1 = []
	n2 = []
	mu2 = []
	vrtt2 = []
	n3 = []
	mu3 = []
	vrtt3 = []
	vrtt = []
	n4 = []
	n5 = []
	mu4 = []
	mu5 = []
	for k in range(1, K + 1):
		vrtt.append(calc_VRTT(k))

		# miss rates
		mu1.append(calc_mu(k, 1, True) * sigma)
		mu2.append(calc_mu(k, 2, True) * sigma)
		mu3.append(calc_mu(k, 3, True) * sigma)
		mu4.append(calc_mu(k, 4, True) * sigma)
		mu5.append(calc_mu(k, 5, True) * sigma)
		
		# miss probabilities
		n1.append(calc_p(k, 1))
		n2.append(calc_p(k, 2))
		n3.append(calc_p(k, 3))
		n4.append(calc_p(k, 4))
		n5.append(calc_p(k, 5))

	# Show the plot
	x = range(K)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# VRTT
	# ax1.scatter(x, vrtt, s=10, c='k', marker="o")
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'VRTT')
	# plt.title(r'\textbf{VRTT Characterization')

	# Miss rates
	# ax1.scatter(x, mu1, s=10, c='g', marker="o", label=r'$\mu_1$')
	# ax1.scatter(x, mu2, s=10, c='r', marker="o", label=r'$\mu_2$')
	# ax1.scatter(x, mu3, s=10, c='b', marker="o", label=r'$\mu_3$')
	# ax1.scatter(x, mu4, s=10, c='m', marker="o", label=r'$\mu_4$')
	# ax1.scatter(x, mu5, s=10, c='k', marker="o", label=r'$\mu_5$')
	# ax1.legend(loc=0, scatterpoints = 5)
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'Miss Rate $\mu$ (chunks/s)')
	# plt.title(r'\textbf{Miss Rate Characterization')

	# Miss probabilities
	ax1.scatter(x, n1, s=10, c='g', marker="o", label=r'$p_k(1)$')
	ax1.scatter(x, n2, s=10, c='r', marker="o", label=r'$p_k(2)$')
	ax1.scatter(x, n3, s=10, c='b', marker="o", label=r'$p_k(3)$')
	ax1.scatter(x, n4, s=10, c='m', marker="o", label=r'$p_k(4)$')
	ax1.scatter(x, n5, s=10, c='k', marker="o", label=r'$p_k(5)$')
	ax1.legend(loc=0, scatterpoints = 5)
	plt.xlabel(r'Popularity Class $k$')
	plt.ylabel(r'Miss Probability $p_k(i)$')
	plt.title(r'\textbf{Miss Probability Characterization')

	# Render the figure for display...
	plt.show()

def init_10_1():
	''' Copy of 2, but trying to get the expected time in better shape...
	'''
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	# M = 20000
	# K = 10000
	M = 20000
	K = 400
	alpha = 1.6
	lmbda = 40.0
	W = 1.0
	sigma = 10.0       # 10kB chunk size for everything
	x = 200 * sigma
	c = 1
	N = 10 # levels of the binary tree
	rts = 2.0 # 2ms round trip time between nodes
	delta = 10.0 

	# Generate the data sets for each specific class of content items...
	sigma_k = []
	lmbda_k = []
	for i in range(K):
		sigma_k.append(sigma)
		lmbda_k.append(lmbda * calc_q(i + 1)) # see the paper for details
		# lmbda_k.append(expovariate(lmbda * calc_q(i + 1)))
		# lmbda_k.append(np.random.poisson(lmbda * calc_q(i + 1), 1))

	# Do the calculations
	n1 = []
	mu1 = []
	vrtt1 = []
	n2 = []
	mu2 = []
	vrtt2 = []
	n3 = []
	mu3 = []
	vrtt3 = []
	vrtt = []
	n4 = []
	n5 = []
	mu4 = []
	mu5 = []
	for k in range(1, K + 1):
		vrtt.append(calc_VRTT(k))

		# miss rates
		mu1.append(calc_mu(k, 1, True) * sigma)
		mu2.append(calc_mu(k, 2, True) * sigma)
		mu3.append(calc_mu(k, 3, True) * sigma)
		mu4.append(calc_mu(k, 4, True) * sigma)
		mu5.append(calc_mu(k, 5, True) * sigma)
		
		# miss probabilities
		n1.append(calc_p(k, 1))
		n2.append(calc_p(k, 2))
		n3.append(calc_p(k, 3))
		n4.append(calc_p(k, 4))
		n5.append(calc_p(k, 5))

	# Show the plot
	x = range(K)
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	# VRTT
	# ax1.scatter(x, vrtt, s=10, c='k', marker="o")
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'VRTT')
	# plt.title(r'\textbf{VRTT Characterization')

	# Miss rates
	# ax1.scatter(x, mu1, s=10, c='g', marker="o", label=r'$\mu_1$')
	# ax1.scatter(x, mu2, s=10, c='r', marker="o", label=r'$\mu_2$')
	# ax1.scatter(x, mu3, s=10, c='b', marker="o", label=r'$\mu_3$')
	# ax1.scatter(x, mu4, s=10, c='m', marker="o", label=r'$\mu_4$')
	# ax1.scatter(x, mu5, s=10, c='k', marker="o", label=r'$\mu_5$')
	# ax1.legend(loc=0, scatterpoints = 5)
	# plt.xlabel(r'Popularity Class $k$')
	# plt.ylabel(r'Miss Rate $\mu$ (chunks/s)')
	# plt.title(r'\textbf{Miss Rate Characterization')

	# Miss probabilities
	ax1.scatter(x, n1, s=10, c='g', marker="o", label=r'$p_k(1)$')
	ax1.scatter(x, n2, s=10, c='r', marker="o", label=r'$p_k(2)$')
	ax1.scatter(x, n3, s=10, c='b', marker="o", label=r'$p_k(3)$')
	ax1.scatter(x, n4, s=10, c='m', marker="o", label=r'$p_k(4)$')
	ax1.scatter(x, n5, s=10, c='k', marker="o", label=r'$p_k(5)$')
	ax1.legend(loc=0, scatterpoints = 5)
	plt.xlabel(r'Popularity Class $k$')
	plt.ylabel(r'Miss Probability $p_k(i)$')
	plt.title(r'\textbf{Miss Probability Characterization')

	# Render the figure for display...
	plt.show()


def main():
	global N, M, rts, K, alpha, c, x, sigma, sigma_k, lmbda, lmbda_k, delta 

	# Toggle which version to run
	init_5_1()
	# init_5_1_uniform()
	# init_10_1()

if __name__ == "__main__":
	main()