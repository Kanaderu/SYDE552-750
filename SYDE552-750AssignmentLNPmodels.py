# Peter Duggins
# SYDE 552/750
# Assignment: Linear-Nonlinear Poisson Models (LNP)
# March 22, 2016

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 20

#Generate 50 10-second trials of Poisson spikes
#at 25 spikes/s witha 5ms absolute refractory period.
#To do this, draw samples from an approprpiate ISI distribution.
#I'll implement the 5ms refractory period as using dt=5ms.

def generate_poisson_spikes(T,dt,trials,rate,rng):
	
	spike_raster=[]
	spike_times=[]
	for trial in range(trials):
		spike_raster_trial=[]
		spike_times_trial=[]
		for t in range(int(T/dt)):
			spike_here=(rng.rand()<rate*dt)
			spike_raster_trial.append(1*spike_here)
			if spike_here:
				spike_times_trial.append(t*dt)
		spike_raster.append(spike_raster_trial)
		spike_times.append(spike_times_trial)
	return np.array(spike_raster),np.array(spike_times)

def get_ISI(spike_times):

	#calculate across all trials
	ISI=[]
	for trial in range(len(spike_times)):
		for t in range(len(spike_times[trial])-1):
			ISI.append(spike_times[trial][t+1]-spike_times[trial][t])
	return ISI

def get_CV(ISI):

	return np.std(ISI)/np.average(ISI)

def get_fano_factor(spike_raster,t_range):

	count=[np.sum(raster[:t_range]) for raster in spike_raster]
	avg=np.average(count)
	if avg != 0:
		var=np.var(count)
	else:
		var=0
	return var/avg

def get_fano_factor_list(T,dt,trials,rate,rng,n_FFs,t_range):

	FF_list=[]
	for i in range(n_FFs):
		spike_raster, spike_times = generate_poisson_spikes(T,dt,trials,rate,rng)
		FF_i=get_fano_factor(spike_raster,t_range)
		FF_list.append(FF_i)
	return FF_list

def one():

	T=10 #seconds
	dt=0.005
	trials=50
	rate=25 #Hz
	seed=3
	t=np.arange(0,T,dt)
	rng=np.random.RandomState(seed=seed)

	spike_raster, spike_times = generate_poisson_spikes(T,dt,trials,rate,rng)

	ISI = get_ISI(spike_times)
	n_bins=int(np.max(ISI)/(2*dt))

	#Plot the spike raster for first 1.0 seconds
	fig=plt.figure(figsize=(16,16))
	ax=fig.add_subplot(211)
	ax.eventplot(spike_times,colors=[[0,0,0]])
	ax.set_xlim(0,1.0)
	ax.set_ylim(0,trials)
	ax.set_xlabel('time (s)')
	ax.set_ylabel('neuron')

	#plot ISI histogram
	ax=fig.add_subplot(212)
	ax.hist(ISI,n_bins)
	ax.set_xlim(0,dt*100)
	ax.set_xlabel('ISI (s)')
	ax.set_ylabel('frequency')
	plt.show()

	CV = get_CV(ISI)
	print "The coefficient of variation is", CV

	t_range=int(0.100/dt)
	n_FFs = 50

	FF_list_1 = get_fano_factor_list(T,dt,trials,rate,rng,n_FFs,t_range)
	print "Fano Factor for $t_{ref}=%s$, %s trials:" %(dt, n_FFs)
	print "mean: %s" %np.average(FF_list_1), "std: %s" %np.std(FF_list_1)

	dt=0.001
	FF_list_2 = get_fano_factor_list(T,dt,trials,rate,rng,n_FFs,t_range)
	print "Fano Factor for $t_{ref}=%s$, %s trials:" %(dt, n_FFs)
	print "mean: %s" %np.average(FF_list_2), "std: %s" %np.std(FF_list_2)

	#the mean approaches 1 as t_{ref} appoaches 0, but the std increases slightly

one()