#
#  NAME
#    problem_set2_solutions.py
#
#  DESCRIPTION
#    Open, view, and analyze action potentials recorded during a behavioral
#    task.  In Problem Set 2, you will write create and test your own code to
#    create tuning curves.
#

#Helper code to import some functions we will use
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from scipy import optimize
from scipy import stats


def load_experiment(filename):
    """
    load_experiment takes the file name and reads in the data.  It returns a
    two-dimensional array, with the first column containing the direction of
    motion for the trial, and the second column giving you the time the
    animal began movement during thaht trial.
    """
    data = np.load(filename)[()];
    return np.array(data)

def load_neuraldata(filename):
    """
    load_neuraldata takes the file name and reads in the data for that neuron.
    It returns an arary of spike times.
    """
    data = np.load(filename)[()];
    return np.array(data)
    
def bin_spikes(trials, spk_times, time_bin):
    """
    bin_spikes takes the trials array (with directions and times) and the spk_times
    array with spike times and returns the average firing rate for each of the
    eight directions of motion, as calculated within a time_bin before and after
    the trial time (time_bin should be given in seconds).  For example,
    time_bin = .1 will count the spikes from 100ms before to 100ms after the 
    trial began.
    
    dir_rates should be an 8x2 array with the first column containing the directions
    (in degrees from 0-360) and the second column containing the average firing rate
    for each direction
    """
    # Output of function. the first column contains the directions (in degrees) and 
    # the second column containing the average firing rate for that direction
    dir_rates = [ ]   
    
    # determines the directions of the motions performed by the monkey
    direction_of_motions = np.unique( trials[:,0] )    
    
    # holds the number of spikes detected by trial
    spikes_per_trial = np.zeros( len( trials ) )
    
     # holds the average rate of spikes / s
    avg_firing_rate = np.zeros(8)      
    
    # count the number of spikes per trial 
    # We will count all the spikes that occur from bin_size before to 
    # bin_size after the animal started the movement
    for i in range( len(trials) ):
        # select all spikes that occur after the animal started to move
        spikes_after = spk_times <=  trials[i,1] + time_bin
        # select all spikes that occur before the animal started to move
        spikes_before = spk_times >=  trials[i,1] - time_bin
        # select all spikes that fall in the interval before and after the 
        # animal started to move
        spikes = np.logical_and( spikes_before, spikes_after )
        # count the number of spikes for the respective trial
        spikes_per_trial[i] = len( spk_times[ spikes ]  )
    
    # Group the trials by direction of motion.
    for m in range( len(direction_of_motions) ):
        
        # group trials and spikes by direction of motion
        motion_indx = plt.find( direction_of_motions[m] == trials[:,0] )
        trials_per_motion = trials[ motion_indx, 1 ]
        spikes_per_motion = spikes_per_trial[motion_indx]
        
        # compute the average of the trials and spiker per motions
        avg_trials_per_motion = np.average( trials_per_motion )
        avg_spikes_per_motion = np.average( spikes_per_motion )
        
        # Convert from spike counts to firing rates (spikes/s)
        avg_firing_rate[ m ] = convert_count_spikes_to_rate(avg_spikes_per_motion,
                                                             avg_trials_per_motion, trials)
    
    # create the final output vector
    dir_rates = np.column_stack( (direction_of_motions, avg_firing_rate) )    
    
    return dir_rates

# Convert from spike counts to firing rate (spikes/s). This is a more standard 
# way to present the data. This allows for interpretation independent of bin size.
def convert_count_spikes_to_rate( num_spikes, num_trials, total_trials ):
    
    return ( num_spikes / num_trials )*np.count_nonzero( total_trials )*10 
    
   
def plot_tuning_curves(direction_rates, title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates) and plots a histogram and 
    polar representation of the tuning curve. It adds the given title.
    """
    
    # computes the width of the bars to display
    bin_size = direction_rates[1,0] - direction_rates[0,0]
  
    # histogram
    plt.subplot(2,2,1)  
    
    # plot the histogram
    plt.bar( direction_rates[:,0],direction_rates[:,1],  width = bin_size, align='center' )
    
    plt.xticks( np.arange(0, 361, bin_size) )       # specify the range of ticks
    plt.xlabel('Direction of Motion (degrees)')     # add label to x-axis
    plt.ylabel('Firering Rate (spikes/s)')          # add label to y-axis
    plt.title( 'Histogram: ' +  title)              # add title to histogram
    
    # polar representation
    plt.subplot(2,2,2,polar=True)
    
    spikeFiringRates = direction_rates[:,1]  # select the spikes firing rates     
   
    # the polar representation is like a circle, so we correct the data
    # in a way that the last element of the array if the same as the first one
    spikeFiringRates = np.append(spikeFiringRates, spikeFiringRates[0])
    
    # compute the angles . A circle ranges from 0 to 360 deg
    theta = np.deg2rad( np.arange( 0, 361, bin_size ) )
    
    # plot the polar graph
    plt.polar( theta, spikeFiringRates, label='Firing Rate (spikes/s)' )
    
    plt.legend(loc=8)                   # specify the location of the legend
    plt.title('Polar: ' + title)        # add title to the graph
    
def roll_axes(direction_rates):
    """
    roll_axes takes the x-values (directions) and y-values (direction_rates)
    and return new x and y values that have been "rolled" to put the maximum
    direction_rate in the center of the curve. The first and last y-value in the
    returned list should be set to be the same. (See problem set directions)
    Hint: Use np.roll()
    """
   
    
    return new_xs, new_ys, roll_degrees    
    

def normal_fit(x,mu, sigma, A):
    """
    This creates a normal curve over the values in x with mean mu and
    variance sigma.  It is scaled up to height A.
    """
    n = A*mlab.normpdf(x,mu,sigma)
    return n

def fit_tuning_curve(centered_x,centered_y):
    """
    This takes our rolled curve, generates the guesses for the fit function,
    and runs the fit.  It returns the parameters to generate the curve.
    """

    return p
    


def plot_fits(direction_rates,fit_curve,title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates and fit_curve) and plots the 
    actual values with circles, and the curves as lines in both linear and 
    polar plots.
    """
    

def von_mises_fitfunc(x, A, kappa, l, s):
    """
    This creates a scaled Von Mises distrubition.
    """
    return A*stats.vonmises.pdf(x, kappa, loc=l, scale=s)


    
def preferred_direction(fit_curve):
    """
    The function takes a 2-dimensional array with the x-values of the fit curve
    in the first column and the y-values of the fit curve in the second.  
    It returns the preferred direction of the neuron (in degrees).
    """
  
    return pd
    
        
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    trials = load_experiment('trials.npy')   
    spk_times = load_neuraldata('example_spikes.npy') 
    
    time_bin = 0.1          # counts the spikes from 100ms before to 100ms after the trial began
    dir_rates = bin_spikes(trials, spk_times, time_bin)
    
    plot_tuning_curves(dir_rates, 'Example Neuron Tunning Curve')

