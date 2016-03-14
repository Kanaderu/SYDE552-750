# Peter Duggins
# SYDE 552/750
# Assignment: Neuron Responses
# March 15, 2016

#Part 1: Tuning Curves
import numpy as np
import scipy.signal
import pickle
import matplotlib.pyplot as plt
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['font.size'] = 20

def load_data_one():

	tuning_data=pickle.load(open('MT-direction-tuning.pkl','rb'))
	directions=tuning_data['direction']
	spikeTimes=tuning_data['spikeTimes']
	return directions,spikeTimes

def one_b():

	#Load the synthetic data file MT-tuning-direction
	directions,spikeTimes=load_data_one()

	#Get the indices of the trials with 0 degree stimulus direction
	zero_degree_indices=np.where(directions==0)

	#Get the spike timing data from those trials
	ugly_arrays=spikeTimes[zero_degree_indices]
	zero_degree_spike_trials=np.array([ugly_arrays[i].flatten()
		for i in range(len(ugly_arrays))])

	#Calculate the 'multi-trial' (or 'time-varying') firing rate
	#by counting the number of spikes in a small time window
	#accross all trials
	T=2.0 #seconds
	bin_width=0.005
	multitrial_binned_rate=[]
	for i in range(int(T/bin_width)):
		bin_i=0
		for trial in zero_degree_spike_trials:
			for t in trial:
				bin_i+=(i*bin_width<=t<(i+1)*bin_width)
		multitrial_binned_rate.append(bin_i)

	#plot spike raster and multitrial firing rate
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(211)
	ax.eventplot(zero_degree_spike_trials,colors=[[0,0,0]])
	ax.set_xlim(0,T)
	ax.set_xlabel('time')
	ax.set_ylabel('neuron')
	ax=fig.add_subplot(212)
	ax.bar(np.arange(0,T,bin_width),multitrial_binned_rate,width=bin_width)
	ax.set_xlim(0,T)
	ax.set_xlabel('time')
	ax.set_ylabel('multi-trial binned spike rate')
	plt.show()

def one_c():

	#Load the synthetic data file MT-tuning-direction
	directions,spikeTimes=load_data_one()
	T=2.0 #seconds
	dt=0.005
	t=np.arange(0,T,dt)
	Nt=len(t)

	#Get the data from trial 9
	trial9_spikes=spikeTimes[0][8].flatten()

	#Create an array that has 1s at the spike times and zeros elsewhere
	trial9_raster=np.zeros((Nt))
	for spike in trial9_spikes:
		trial9_raster[spike/dt] = 1

	#Define the smoothing Gaussian kernels
	sigma1=0.005
	sigma2=0.05
	G1 = np.exp(-(t-np.average(t))**2/(2*sigma1**2))     
	G1 = G1 / sum(G1)  #normalize
	G2 = np.exp(-(t-np.average(t))**2/(2*sigma2**2))     
	G2 = G2 / sum(G2)  #normalize

	#Convolve Gaussians with the spikes to calculate single-trial rate estimate
	trial9_smoothed1=np.convolve(trial9_raster,G1,'same')
	trial9_smoothed2=np.convolve(trial9_raster,G2,'same')
	# trial9_smoothed3=gaussian_filter(trial9_raster,sigma1)
	# trial9_smoothed4=gaussian_filter(trial9_raster,sigma2)

	#Plot the single-trial rate estimates
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(t,trial9_smoothed1,label='$\\sigma=%s$' %sigma1)
	ax.plot(t,trial9_smoothed2,label='$\\sigma=%s$' %sigma2)
	# ax.plot(t,trial9_smoothed3,label='$\\sigma=%s$' %sigma1)
	# ax.plot(t,trial9_smoothed4,label='$\\sigma=%s$' %sigma2)
	# ax.set_xlim(0,T)
	ax.set_xlabel('time')
	ax.set_ylabel('single-trial rate estimate')
	legend=ax.legend(loc='best',shadow=True)
	plt.show()

def one_d():

	#Load the synthetic data file MT-tuning-direction
	directions,spikeTimes=load_data_one()
	T=2.0 #seconds

	#find the unique direction values in the directions array
	unique_directions=np.unique(directions)

	#for each unique direction, find the trial indices in that direction
	trial_indices=np.array([np.where(directions==u)[1].tolist()
		for u in unique_directions])

	#calculate spike rate = spike count/time for each direction, avg over trials
	rate_vs_direction_mean=[]
	rate_vs_direction_std=[]
	for direction in trial_indices:
		dir_spikes_count=[]
		for trial in direction:
			trial_spike_times=spikeTimes[0][trial][0]
			#find indices of spikes between 50 and 250 ms
			fifty_to_twofifty_indices=np.where(trial_spike_times[
				(0.050<=trial_spike_times) & (trial_spike_times<=0.250)])[0]
			fifty_to_twofifty_spike_count=len(fifty_to_twofifty_indices)
			dir_spikes_count.append(fifty_to_twofifty_spike_count)
		rate_vs_direction_mean.append(np.average(dir_spikes_count))
		rate_vs_direction_std.append(np.std(dir_spikes_count))

	#Plot the tuning curve
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(unique_directions,rate_vs_direction_mean)
	ax.fill_between(unique_directions,
		np.subtract(rate_vs_direction_mean,rate_vs_direction_std),
		np.add(rate_vs_direction_mean,rate_vs_direction_std),
		color='lightgray')
	ax.set_xlabel('angle (degrees)')
	ax.set_ylabel('trial-averaged firing rate, 50-250ms')
	plt.show()

def load_data_two():

	spiking_data=pickle.load(open('c1p8.pkl','rb'))
	stim=spiking_data['stim']
	rho=spiking_data['rho']
	return stim,rho

def spike_trig_avg(stim,spikes,dt,window_width):

	spike_indices=np.where(spikes==1)[0].flatten()
	#calculate the spike-triggered average
	window = int(window_width / dt)
	spike_triggered_avg=[]
	#ignore time points before the first window
	for i in range(len(spike_indices)):
		stim_sum_i=0
		if i > window:
			for j in range(window):
				stim_sum_i+=stim[i-j]
		spike_triggered_avg.append(stim_sum_i)

	spike_triggered_avg=np.array(spike_triggered_avg).flatten()/len(spike_indices)

	#Plot the spike-triggered average
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(spike_indices*dt,spike_triggered_avg,
		label='$\\tau_{window}=%s (s)' %window_width)
	ax.set_xlabel('time (seconds)')
	ax.set_ylabel('spike-triggered average')
	plt.show()

#for each timestep in the window, find the value of the stimuli at time=t 
#before each spike, and append to the list the average of this value over all spikes
#produces array.shape=(window,1)
def spike_trig_avg2(stim,spikes,dt,window_width):

	window = np.arange(0,int(window_width / dt),1)
	#truncate spikes in first window timesteps
	spike_indices=np.where(spikes[len(window):]==1)[0].flatten()
	spike_triggered_avg=[]
	for t in window:
		stim_sum_i=[]
		for i in spike_indices:
			#undo truncation when indexing from stimulus
			stim_sum_i.append(stim[(i+len(window))-t])
		spike_triggered_avg.append(np.average(stim_sum_i))

	spike_triggered_avg=np.array(spike_triggered_avg).flatten()/len(spike_indices)

	return -1.0*window*dt, spike_triggered_avg

def two_b():

	#load the synthetic data
	stim,rho=load_data_two()
	dt=0.002
	window_width=0.200

	#calculate the spike triggered average
	window, sta = spike_trig_avg2(stim,rho,dt,window_width)
	
	#Plot the spike-triggered average
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(window,sta)
	ax.set_xlabel('time before spike (seconds)')
	ax.set_ylabel('spike-triggered average')
	plt.show()

def white_noise(mean=0,std=1,T=100,dt=0.001,rng=np.random.RandomState()):
	return rng.normal(mean,std,T/dt)

def synthetic_neuron(drive):
	"""
	Simulates a mock neuron with a time step of 1ms.
	Arguments:
	drive - input to the neuron (expect zero mean; SD=1)
	Returns:
	rho - response function (0=non-spike and 1=spike at each time step)
	"""	
	  
	dt = 0.001
	T = dt*len(drive)
	time = np.arange(0, T, dt)
	lagSteps = 0.02/dt
	drive = np.concatenate((np.zeros(lagSteps), drive[lagSteps:]))
	system = scipy.signal.lti([1], [0.03**2, 2*0.03, 1])
	_, L, _ = scipy.signal.lsim(system, drive[:,np.newaxis], time)
	rate = np.divide(30, 1 + np.exp(50*(0.05-L)))
	spikeProb = rate*dt
	return np.random.rand(len(spikeProb)) < spikeProb

def two_c():

	T=100
	dt=0.001
	mean=0
	std=1
	seed=3

	#generate noisy signal with gaussian sampled numbers
	rng=np.random.RandomState(seed=seed)
	noise=white_noise(mean,std,T,dt,rng)

	#use Bryan's code to get the spikes from an input signal
	spikes=synthetic_neuron(noise)

	#calculate the spike-triggered average
	window_width=0.200
	window, sta = spike_trig_avg2(noise,spikes,dt,window_width)

	#Plot the spike-triggered average
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(window,sta)
	ax.set_xlabel('time (seconds)')
	ax.set_ylabel('spike-triggered average')
	plt.show()

def two_d():

	T=100
	dt=0.001
	mean=0
	std=1
	seed=3

	#generate noisy signal with gaussian sampled numbers
	rng=np.random.RandomState(seed=seed)
	noise=white_noise(mean,std,T,dt,rng)

	#generate colored noise by convolving the noise signal with a gaussian
	t=np.arange(0,T,dt)
	sigma=0.020
	G = np.exp(-(t-np.average(t))**2/(2*sigma**2))     
	G = G / sum(G)
	colored_noise=np.convolve(noise,G,'same')

	#feed colored noise into Bryan's spike generator
	spikes=synthetic_neuron(colored_noise)

	#calculate the spike-triggered average
	window_width=0.200
	window, sta = spike_trig_avg2(colored_noise,spikes,dt,window_width)

	#Plot the spike-triggered average
	fig=plt.figure(figsize=(16,8))
	ax=fig.add_subplot(111)
	ax.plot(window,sta)
	ax.set_xlabel('time (seconds)')
	ax.set_ylabel('spike-triggered average')
	plt.show()

def main():

	# one_b()
	# one_c()
	# one_d()
	# two_b()
	two_c()
	# two_d()

main()