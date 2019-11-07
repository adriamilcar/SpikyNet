import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


# Population of Leaky Integrate-and-Fire (LIF), Resonate-and-Fire (RF), or Izhikevich (IZ) neurons.
class Population(object):

    # Initializes a LIF population with all the specified parameters, without weight matrix.
    def __init__(self, N, type, tau_m, C, R, tau_ref, Vth, Vrest, Vreset, w, b, dt):

        self.dt = dt                                      # Simulation time step (ms).
        self.N = N                                        # Number of neurons.
        self.C = C                                        # Neuron's capacitance (pF).
        self.R = R                                        # Neuron's membrane resistance (MOhm).
        self.tau_m = tau_m
        if tau_m == 0:
            self.tau_m = self.R*self.C                    # Membrane time constant (ms).
        self.tau_ref = tau_ref                            # Refractory period (ms).
        self.Vth = Vth                                    # Spike threshold (mV).
        self.Vrest = Vrest                                # Resting voltage at steady-state (mV).
        self.Vreset = Vreset                              # Voltage at which a neuron resets after a spike (mV).
        self.Vm = np.zeros(self.N) + Vrest                # Membrane potential (mV) trace over time.
        self.last_spike = np.zeros(self.N) - 1e100        # Array of times for each neuron's last spike event (ms).
        self.raster = [[] for _ in np.arange(self.N)]     # Raster of spikes for all neurons.
        self.spiked = np.zeros(self.N, dtype=bool)        # Mask for neurons that spiked in the last time step.
        self.type = type                                  # LIF, RF, or Izhikevich neurons.

        if self.type == 'RF':
            self.Vm = self.Vm*1j
            self.Vrest = self.Vrest*1j
            self.Vreset = self.Vreset*1j

            self.w = w                                    # Intrinsic oscillatory frequency of membrane voltage.
            self.b = b                                    # Rate of converge/divergence to/from the reset point.
            self.STO = (self.b - 1j*self.w)               # Sub-threshold oscillations.

    # The population advances one iteration.
    def activate(self, t, I):

        if self.type == 'LIF':
            self.Vm[self.spiked] = self.Vreset
            active = t > (self.last_spike + self.tau_ref)     # Changes in voltage only for the ones out of the refractory period.
            self.Vm[active] += ((I[active]*self.R)/self.tau_m + (self.Vrest-self.Vm[active])/self.tau_m) * self.dt  # Integration and decay.
            self.spiked = self.Vm > self.Vth                  # Only spike the ones that surpass the threshold.

        elif self.type == 'RF':
            self.Vm[self.spiked] = self.Vreset
            active = t > (self.last_spike + self.tau_ref)
            self.Vm[active] += ((I[active]*self.R)/self.tau_m*1j + (self.Vrest-self.Vm[active])/self.tau_m*self.STO) * self.dt
            self.spiked = np.imag(self.Vm) > self.Vth

        elif self.type == 'IZ':
            pass

        self.last_spike[self.spiked] = t
        for i in np.arange(self.N)[self.spiked]:
            self.raster[i].append(t)

        return self.last_spike

    # Resets Population activity.
    def reset(self):
        self.raster = [[] for _ in np.arange(self.N)]
        self.spiked = np.zeros(self.N, dtype=bool)
        self.last_spike = np.zeros(self.N) - 1e100
        if self.type == 'LIF':
            self.Vm = np.zeros(self.N) + self.Vrest
        elif self.type == 'RF':
            self.Vm = (np.zeros(self.N) + np.imag(self.Vrest)) * 1j


# Connectivity between two populations of LIF neurons.
class Synapses(object):

    # Initializes the connectivity matrices and logs for 2 interconnected populations of LIF neurons.
    def __init__(self, N_pre, N_post, PrePost_ratio, reverse, ordered, sign, g_syn, tau_psc, tau_m, learning, mean, std, A, bias):

        self.N_pre = N_pre                       # Number of neurons of presynaptic population.
        self.N_post = N_post                     # Number of neurons of postsynaptic population.
        self.ratio = PrePost_ratio               # Pre-Post convergence ratio (number of presynaptic neurons that synpase each postsynaptic neuron).
        self.reverse = reverse                   # Reverse convergence ratio to 'divergence ratio' (how many postsynaptic neurons are targeted by each presynaptic neuron).
        self.ordered = ordered
        self.sign = sign                         # +1 for excitatory synapses and -1 for inhibitory ones.
        self.mean = mean
        self.std = std
        self.conns = np.zeros((self.N_pre, self.N_post))
        self.synapses = self.build_synapses()
        self.learning = learning                 # Indicates if change of weights will take place, and direction.
        self.I_syn = np.zeros(N_post)            # Current synaptic activity over time.
        self.g_syn = g_syn                       # Peak current (nS) achieved by an incoming spike over the synapse.
        self.tau_psc = tau_psc                   # Post synaptic current filter time constant (decay of peak current).
        self.tau_m = tau_m                       # Time constant (msec).
        self.Ap, self.Ad = A                     # Parameters balancing LTP and LTD.
        self.bias = bias                         # Time bias of learning rule ("optimal delay").

    # The synapses between neurons of both populations are randomly build according to the *absolute* ratio.
    def build_synapses(self):

        N1 = self.N_pre
        N2 = self.N_post

        if self.reverse:
    	    N1 = self.N_post
    	    N2 = self.N_pre

        synapses = np.zeros((N1, N2))

        if self.ordered:   # build topographically ordered synapses.
            indxs = np.arange(0, N1, self.ratio)
            for i, indx in enumerate(indxs):
        	    synapses[indx:indx+self.ratio, i] = np.abs(np.random.normal(self.mean, self.std)/self.ratio) * self.sign
        else:
	        for i in np.arange(N2):
	            indxs = np.random.choice(N1, self.ratio, replace=False)
	            for indx in indxs:
	                synapses[indx, i] = np.abs(np.random.normal(self.mean, self.std)/self.ratio) * self.sign  # Normalized weights.

        self.conns = np.copy(synapses)
        self.conns[np.nonzero(self.conns)] = 1

        if self.reverse:
        	self.conns = self.conns.T
        	synapses = synapses.T

        return synapses

    # Synaptic model.
    def Isyn(self, t):

        t[t < 0] = 0    # t is an array of times since each neuron's last spike event.
        return self.g_syn * np.exp(-t/self.tau_psc)  # Exponential decay.
        #return t * np.exp(-t/self.tau_psc) * 1e-7  # Exponential decay.

    # Generate synaptic currents for postsynaptic neurons.
    def activate(self, t, pre_last_spike):

        self.I_syn = np.dot(self.synapses.T, self.Isyn(t - pre_last_spike)) 
        return self.I_syn

    # Updates the weights based on the STDP (and anti-STDP) rule.
    def update(self, t, pre_last_spike, post_last_spike):

        pre = pre_last_spike == t
        post = post_last_spike == t
        delta = np.zeros((self.N_pre, self.N_post))

        if t in post_last_spike:
            delta[:, post] = np.tile(self.Ap*np.exp(-(t-pre_last_spike)/self.tau_m) * self.sign * self.learning, (np.count_nonzero(post),1)).T

        if t in pre_last_spike:
            delta[pre, :] = np.tile(-self.Ad*np.exp((post_last_spike-t)/self.tau_m) * self.sign * self.learning, (np.count_nonzero(pre),1))

        if np.any(pre) and np.any(post):
            delta[np.ix_(pre, post)] = 0.

        delta *= self.conns
        self.synapses += delta

        # Keep the synapses either excitatory or inhibitory.
        if self.sign > 0:
            self.synapses[self.synapses < 0.] = 0.
        elif self.sign < 0:
            self.synapses[self.synapses > 0.] = 0.

    def reset(self, weights=False):

        self.I_syn = np.zeros(self.N_post)
        if weights:
            self.synapses = self.build_synapses()


# Creates a Circuit with populations of LIF or IZ neurons and plastic synapses among them.
class Model(object):

    # Initializes a circuit according to the provided parameters.
    def __init__(self, dt, verbose=0):

        self.N = 0                       # Number of populations.
        self.N_pop = []                  # Number of neurons of each population (1-D array).
        self.pop_ratios = []             # Absolute connectivity ratios between populations (NxN matrix).
        self.signs = []                  # Defines excitatory or inhibitory synapses (NxN matrix).
        self.learning = []               # Booleans encoding synaptic plasiticy among populations (NxN matrix).
        self.taus_m = []                 # Time constants (msec) for neurons in each population (1-D array).
        self.taus_ref = []               # Refractory periods (msec) for neurons in each population (1-D array).
        self.taus_psc = []               # Post synaptic current filter time constant for synapses (NxN matrix).
        self.Vths = []                   # Spike thresholds for neurons in each population (1-D array).
        self.names = []                  # Names of populations.
        self.I_ext = []                  # External current to each population.
        self.rasters = []

        self.t = 0                       # Simulation time (msec).
        self.dt = dt*1e-3                # Simulation time step (msec).

        self.PP = []                     # 1-D array containing the Populations of neurons.
        self.Synapses = []               # NxN matrix containing the synapses between each pair of Populations.

        self.last_spike = []             # Array of times of last spike for neurons in each population.
        self.verbose = verbose           # Level of online report during simulation.

    # Creates a Population.
    def add_Population(self, name, N=100, type='LIF', tau_m=15, C=150, R=100, tau_ref=4, Vth=-50, Vrest=-70, Vreset=-70, w=2, b=.1):

        self.N += 1
        self.names.append(name)
        self.N_pop.append(N)
        if tau_m == 0:
            self.taus_m.append(R*1e6*C*1e-12)
        else:
            self.taus_m.append(tau_m*1e-3)
        self.taus_ref.append(tau_ref*1e-3)
        self.Vths.append(Vth*1e-3)
        self.PP.append(Population(N, type, tau_m*1e-3, C*1e-12, R*1e6, tau_ref*1e-3, Vth*1e-3, Vrest*1e-3, Vreset*1e-3, w, b, self.dt))
        self.last_spike.append(self.PP[-1].last_spike)
        self.I_ext.append(np.zeros(N))

        new_synapses = np.zeros((self.N, self.N))
        new_synapses[:self.N-1, :self.N-1] = self.Synapses
        self.Synapses = new_synapses.tolist()

        new_ratios = np.zeros((self.N, self.N))
        new_ratios[:self.N-1, :self.N-1] = self.pop_ratios
        self.pop_ratios = new_ratios

        new_signs = np.zeros((self.N, self.N))
        new_signs[:self.N-1, :self.N-1] = self.signs
        self.signs = new_signs

        new_learning = np.zeros((self.N, self.N))
        new_learning[:self.N-1, :self.N-1] = self.learning
        self.learning = new_learning

        new_taus = np.zeros((self.N, self.N))
        new_taus[:self.N-1, :self.N-1] = self.taus_psc
        self.taus_psc = new_taus

    # Creates a Connection between two Populations.
    def add_Connection(self, pre, post, connect_ratio, reverse, sign, learning, ordered, g_syn, tau_psc, mean, std, A=[0.5,0.5], bias=0):

        i = np.where(pre == np.array(self.names))[0][0]
        j = np.where(post == np.array(self.names))[0][0]
        self.pop_ratios[i][j] = connect_ratio
        self.signs[i][j] = sign
        self.learning[i][j] = learning
        self.taus_psc[i][j] = tau_psc*1e-3
        self.Synapses[i][j] = Synapses(N_pre=self.N_pop[i], N_post=self.N_pop[j], PrePost_ratio=connect_ratio, reverse=reverse, ordered=ordered,
                                       sign=sign, g_syn=g_syn*1e-9, tau_psc=tau_psc*1e-3, tau_m=self.taus_m[j], learning=learning, mean=mean, std=std, A=A, bias=bias)

    # Generates a noisy (Gaussian) current into a given Population.
    def add_externalCurrent(self, population, mean, std, indxs=[], I=[], N=0):

        indx = np.where(population == np.array(self.names))[0][0]

        if len(I) > 0:
            self.I_ext[indx] = I

        elif len(indxs) > 0:
            self.I_ext[indx][indxs] = np.random.normal(mean, std, len(indxs))

        elif N > 0:
            self.I_ext[indx][np.random.choice(np.arange(self.N_pop[indx], dtype=int), N)] = np.random.normal(mean, std, N)

        else:
            self.I_ext[indx] = np.random.normal(mean, std, self.N_pop[indx])

    # Reset external currents for all Populations.
    def reset_externalCurrents(self):

        self.I_ext = [I*0. for I in self.I_ext]

    # Advances one time step (dt) for the whole circuit.
    def advance(self):

        if self.t == 0 and self.verbose == 1:
            print('Starting...')

        self.t = np.round(self.t + self.dt, decimals=6)

        if ((self.t%1) == 0.) and (self.verbose == 1):
            print(str(int(self.t)) + ' s')

        new_last_spike = []
        for j in np.arange(self.N):

            I_syn = []
            for i in np.arange(self.N):
                if self.signs[i][j] != 0:
                    I_syn_i = self.Synapses[i][j].activate(self.t, self.last_spike[i])
                    I_syn.append(I_syn_i)

            I_syn = np.sum(I_syn, axis=0)
            I = I_syn + self.I_ext[j]
            spk = self.PP[j].activate(self.t, I)
            new_last_spike.append(spk)

        self.last_spike = new_last_spike

        # The weights are updated for the plastic synapses based on the new spikes.
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if self.learning[i][j] != 0:
                    self.Synapses[i][j].update(self.t, self.last_spike[i], self.last_spike[j])

        self.reset_externalCurrents()

    # Resets all neurons (voltages and last spikes).
    def reset(self, weights=False):

        for i in np.arange(self.N):
            self.PP[i].reset()
            for j in np.arange(self.N):
                if self.signs[i][j] != 0:
                    self.Synapses[i][j].reset(weights=weights)

        self.reset_externalCurrents()
        self.t = 0.

    # Show all information about the Populations and Synapses.
    def summary(self):

        print('\033[1m Populations Information \033[0m')
        info = [self.N_pop, self.taus_m, self.taus_ref, self.Vths]
        names_info = ['Number of Neurons', 'Membrane Taus (s)', 'Refractory Period (s)', 'Spike Threshold (V)']
        display(pd.DataFrame(info, names_info, self.names))
        print('')
        print('')

        print('\033[1m Connectivity Matrix \033[0m')
        conn_m = pd.DataFrame((self.pop_ratios*self.signs).astype(int), self.names, self.names)
        ii, jj = np.nonzero(self.learning)
        conn_m.style.applymap(lambda: 'color: green', subset=pd.IndexSlice[ii, jj])
        display(conn_m)
        print('')
        print('')

        print('\033[1m Synaptic Taus (s) \033[0m')
        display(pd.DataFrame(self.taus_psc, self.names, self.names))
        print('')
        print('')

    # Removes a Population with all its synapses to other populations.
    def remove_Population(self, population):

        i = np.where(population == np.array(self.names))[0][0]

        self.N -= 1

        del self.names[i]
        del self.N_pop[i]
        del self.taus_m[i]
        del self.taus_ref[i]
        del self.Vths[i]
        del self.PP[i]
        del self.last_spike[i]
        del self.I_ext[i]

        self.pop_ratios = np.delete(self.pop_ratios, i, axis=0)
        self.pop_ratios = np.delete(self.pop_ratios, i, axis=1)

        self.signs = np.delete(self.signs, i, axis=0)
        self.signs = np.delete(self.signs, i, axis=1)

        self.learning = np.delete(self.learning, i, axis=0)
        self.learning = np.delete(self.learning, i, axis=1)

        self.taus_psc = np.delete(self.taus_psc, i, axis=0)
        self.taus_psc = np.delete(self.taus_psc, i, axis=1)

        for row in self.Synapses:
            del row[i]

        del self.Synapses[i]
        #for col in np.arange(self.N):
            #del self.Synapses[i][col]
        #del self.Synapses[i]

    # Removes a specific connection of synapses between two populations.
    def remove_Connection(self, pre, post):

        i = np.where(pre == np.array(self.names))[0][0]
        j = np.where(post == np.array(self.names))[0][0]

        self.pop_ratios[i][j] = 0
        self.signs[i][j] = 0
        self.learning[i][j] = 0
        self.taus_psc[i][j] = 0
        self.Synapses[i][j] = 0

    # Removes the plastic feature of synapses between two populations.
    def remove_Plasticity(self, pre, post):

        i = np.where(pre == np.array(self.names))[0][0]
        j = np.where(post == np.array(self.names))[0][0]
        self.Synapses[i][j].learning = 0
        self.learning[i][j] = 0

    # Makes the synapses between pre and post populations plastic (rule: STDP == 1 and anti-STDP == -1).
    def add_Plasticity(self, pre, post, rule):

        i = np.where(pre == np.array(self.names))[0][0]
        j = np.where(post == np.array(self.names))[0][0]
        self.Synapses[i][j].learning = rule
        self.learning[i][j] = rule

    # Readout of membrane voltages for neurons of specific populations (indxs, from 0 to len(N_pop[i])).
    def readout_Population(self, population, indxs=[]):

        i = np.where(population == np.array(self.names))[0][0]

        if len(indxs) == 0:
            return self.PP[i].Vm
        else:
            return np.array(self.PP[i].Vm)[indxs[0]:indxs[1]].tolist()

    # Readout of weights for synapses between specific populations.
    def readout_Weights(self, pre, post):

        i = np.where(pre == np.array(self.names))[0][0]
        j = np.where(post == np.array(self.names))[0][0]

        return self.Synapses[i][j].synapses

    def get_rasters(self):
        self.rasters = [p.raster for p in self.PP]
        return self.rasters

    # Raster plots.
    def plot_rasters(self, time_limits=[], marker_size=5.):

        #self.raster_plots = [p.raster for p in self.PP]
        rasters = self.get_rasters()
        plt.figure(figsize=(15, int(6.5*(self.N/2))), dpi=300)

        for p in range(self.N):

            plt.subplot(int(self.N+1/2), 2, p+1)

            for i in np.arange(self.N_pop[p]):
                plt.scatter(self.rasters[p][i], np.ones(len(self.rasters[p][i])) * (i+1),
                            marker='.', s=marker_size, linewidth=1., c='k')

            plt.title(str(self.names[p]), fontsize=16)
            plt.xticks(np.linspace(0, self.t, 5), np.round(np.linspace(0, self.t, 5, dtype=float), decimals=2), fontsize=14)
            if self.N_pop[p] > 10:
                plt.yticks([1, self.N_pop[p]], [1, self.N_pop[p]], fontsize=14)
            else:
                plt.yticks(np.arange(self.N_pop[p]+1), fontsize=14)
            if len(time_limits) > 0:
                plt.xlim(time_limits[0], time_limits[1])
            plt.ylim(bottom=-0.05*self.N_pop[p]+1, top=1.05*self.N_pop[p])

            if p%2 != 1:
                plt.ylabel('Neuron', fontsize=14)

            if p == self.N-2 or p == self.N-1:
                plt.xlabel('Time (s)', fontsize=14)

        plt.tight_layout()
        plt.show()
        return rasters

    # Plots firing rates for all Populations, raw (normal) or averaged with a sliding time window.
    def plot_firingRates(self, type='normal', time_window=1, time_limits=[]):

        plt.figure(figsize=(15, int(6.5*(self.N/2))), dpi=300)

        rates = []
        for p in np.arange(self.N):

            plt.subplot(int(self.N+1/2), 2, p+1)

            plt.title(str(self.names[p]) + ' (' + str(self.N_pop[p]) + ' neurons)', fontsize=16)

            raster = np.zeros((self.N_pop[p], int((self.t+self.dt)/self.dt)))
            for i in np.arange(self.N_pop[p]):
                spk = (np.array(self.rasters[p][i])/self.dt).astype(int)
                raster[i][spk] = 1.

            rate = []
            if type == 'normal':
                rate = np.sum(raster, axis=0)
            elif type == 'average':
                for t in np.arange(0, int(((self.t+self.dt)/self.dt)-(time_window/self.dt)), 1):
                    rate.append(np.sum(raster[:, t:t+int(time_window/self.dt)], axis=(0, 1)))
            else:
                # raster = raster[:, :-1].reshape((raster.shape[0], int(raster.shape[1]/(time_window/self.dt)), int(time_window/self.dt)))
                # rate = np.sum(raster, axis=(0, 2))
                pass

            rates.append(rate)

            plt.plot(rate, color='black', linewidth=0.5)

            if p%2 != 1:
                plt.ylabel('Pop. firing rate (#spikes)', fontsize=14)

            if p == self.N-2 or p == self.N-1:
                plt.xlabel('Time (s)', fontsize=14)

            plt.xticks(np.linspace(0, int((self.t+self.dt)/self.dt), 5), np.round(np.linspace(0, self.t, 5, dtype=float), decimals=2), fontsize=14)
            plt.yticks(fontsize=14)

            if len(time_limits) != 0:
                plt.xlim(time_limits[0], time_limits[1])

            plt.ylim(bottom=-0.05*np.amax(rate), top=1.05*np.amax(rate))

        plt.tight_layout()
        plt.show()

    # Plots time-frequency analysis.
    def plot_timeFreq(self):
        pass
        '''from mne.time_frequency import tfr_array_morlet
        from mne.baseline import rescale

        # https://www.martinos.org/mne/stable/auto_examples/time_frequency/plot_time_frequency_simulated.html?highlight=multitaper


        freqs = np.arange(5., 100., 3.)
        vmin, vmax = -3., 3.
        n_cycles = freqs / 2.
        time_bandwidth = 2.0

        power = tfr_array_morlet(rates[0].reshape((1,1,rates[0].shape[0])), sfreq=1000.,
                                 freqs=freqs, n_cycles=n_cycles,
                                 output='avg_power')
        # Baseline the output
        #rescale(power, len(rates[0]), (0., 0.1), mode='mean', copy=False)
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(len(rates[0]) * 1000, freqs, power[0],
                             cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title('TFR calculated on a numpy array')
        ax.set(ylim=freqs[[0, -1]], xlabel='Time (ms)')
        fig.colorbar(mesh)
        plt.tight_layout()

        plt.show()'''
