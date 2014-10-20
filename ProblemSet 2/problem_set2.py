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
    
    # count the number of spikes per trial 
    # We will count all the spikes that occur from bin_size before to 
    # bin_size after the animal started the movement
    spikes_per_trial = count_spikes_per_trial( spk_times, trials, time_bin  )
    
    # Group the trials by direction of motion & compute the average of the trials and spiker per motion
    avg_firing_rate = compute_firing_rate_per_motion( direction_of_motions, spikes_per_trial, trials )
    
    # create direction rates vector
    dir_rates = np.column_stack( (direction_of_motions, avg_firing_rate) )    
    
    # start the fitting process
    # we need to fit the data in a normal distribution
    # in order todo so, we need to change the direction rates in such a way,
    # that the maximum peak of the function is aligned with the center of the 
    # histogram

    # we will perform a shift on the data in order to put the maximum peak in 
    # the middle of the histogram, just like in a normal distribution
    new_xs, new_ys, degrees = roll_axes( dir_rates )
    
    # next, we will try to fit the normal distribution in the data
    fitting_curve = fitting( new_xs, new_ys, dir_rates, degrees, 'Tuning Curve Fit' )
    
    # shift the data back
    new_xs, new_ys = roll_back( fitting_curve, degrees*45 )  
   
    fitting_curve = np.column_stack( (new_xs, new_ys ) )   
    
    return dir_rates, fitting_curve
    
    #return dir_rates

# Group the trials by direction of motion & compute the average of the trials and spiker per motion
def compute_firing_rate_per_motion( direction_of_motions, spikes_per_trial, trials ):
    
    # holds the average rate of spikes / s
    avg_firing_rate = np.zeros(8)  
    
    # Group the trials by direction of motion.
    for m in range( len( direction_of_motions ) ):
        
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
    return avg_firing_rate

# count the number of spikes per trial 
# We will count all the spikes that occur from bin_size before to 
# bin_size after the animal started the movement
def count_spikes_per_trial( spk_times, trials, time_bin  ):
    
    # holds the number of spikes detected by trial
    spikes_per_trial = np.zeros( len( trials ) )
    
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
        
    return spikes_per_trial

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
    
    min_x = np.min( direction_rates[:,0] )-22
    max_x = np.max( direction_rates[:,0] )+22
    
    min_y = np.min( direction_rates[:,1] )
    max_y = np.max( direction_rates[:,1] )+2
    
    # plot the histogram
    plt.bar( direction_rates[:,0], direction_rates[:,1],  width = bin_size, align='center' )
    plt.xticks( np.arange(0, 361, bin_size) )       # specify the range of ticks
    plt.xlim( ( min_x,  max_x ) )
    plt.ylim( ( min_y,  max_y) )
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
    bin_size = direction_rates[1,0] - direction_rates[0,0]   
    
    new_xs = np.append(direction_rates[:,0], direction_rates[-1,0])
    new_ys = np.append(direction_rates[:,1], direction_rates[-1,1])
      
    roll_degrees = plt.find( max( new_ys ) == new_ys ) 
    roll_degrees = roll_degrees[0]
    
    if( roll_degrees < 4 ):
        roll_degrees = np.abs(roll_degrees - 4 )
    else:
        roll_degrees = 3 - roll_degrees   

    new_xs = np.roll( new_xs, roll_degrees )
    new_ys = np.roll( new_ys,  roll_degrees )
      
    new_ys[0] =  new_ys[ -1 ] 

    zero_indx = plt.find( new_xs == 0  )
    zero_indx = zero_indx[0]
   
    for i in range( len(new_xs) ):
        if( i < zero_indx ):
            new_xs[ i ] = ( zero_indx - i )*bin_size*(-1)
          
    #plt.bar( new_xs, new_ys, width=45, align='center')
    #plt.xticks( np.arange( np.min(new_xs) ,  np.max(new_xs)+bin_size , bin_size  ))
    #plt.xlim( ( np.min(new_xs) - bin_size ,  np.max(new_xs) + bin_size)  )
    
    return new_xs, new_ys, roll_degrees    
    

def normal_fit(x,mu, sigma, A):
    """
    This creates a normal curve over the values in x with mean mu and
    variance sigma.  It is scaled up to height A.
    """
    n = A*mlab.normpdf(x,mu,sigma)
    return n

def fit_tuning_curve(centered_x, centered_y):
    """
    This takes our rolled curve, generates the guesses for the fit function,
    and runs the fit.  It returns the parameters to generate the curve.
    """
    
    bin_size = np.abs( centered_x[1] - centered_x[0] )  # size of bar

    max_y = np.amax( centered_y )                       # What is the biggest y-value? (This
                                                        # estimates the amplitude of the curve
    
    max_x = centered_x[ np.argmax( centered_y) ]        # Where is the biggest y-value? (This 
                                                        # estimates the mean of the curve, mu)
   
    var = 2*bin_size                                    # Here we are approximating one standard
                                                        # deviation of our normal distribution
                                                        # (which should be around the width of 2
                                                        # bars)
    
    p, cov = optimize.curve_fit(normal_fit, centered_x, centered_y, p0=[max_x, var, max_y])

    return p
    
def plot_fits(direction_rates,fit_curve,title):
    """
    This function takes the x-values and the y-values  in units of spikes/s 
    (found in the two columns of direction_rates and fit_curve) and plots the 
    actual values with circles, and the curves as lines in both linear and 
    polar plots.
    """
    bin_size = 45
        
    x_axis = direction_rates[:,0]
    y_axis = direction_rates[:,1]    
    
    min_lim = np.min(x_axis)    
    max_lim = np.max(x_axis) + bin_size
    
    plt.subplot(2,2,3)
    plt.plot(x_axis, y_axis,'o',hold=True)
    
    plt.plot( fit_curve[:,0], fit_curve[:,1],'-')
    plt.xlim( (min_lim, max_lim + bin_size) )
    plt.xticks( np.arange( np.min(x_axis), np.max(x_axis) + 2*bin_size, bin_size  ) )
    plt.xlabel('Direction of Motion ( degrees )')
    plt.ylabel('Firering Rate (spikes/s)')
    plt.title( title )
    
    spikeFiringRates = direction_rates[:,1]  # select the spikes firing rates     

    # the polar representation is like a circle, so we correct the data
    # in a way that the last element of the array if the same as the first one
    spikeFiringRates = np.append(spikeFiringRates, spikeFiringRates[0])
   
    # compute the angles . A circle ranges from 0 to 360 deg
    theta = np.deg2rad( np.arange( 0, 361, bin_size ) )
   
    plt.subplot(2,2,4,polar=True)
    # plot the polar graph
    plt.polar( theta, spikeFiringRates, 'o' )
    
    spikeFiringRatesFit = fit_curve[:,1]
    spikeFiringRatesFit = np.append(spikeFiringRatesFit, spikeFiringRatesFit[0])
    
    theta = np.deg2rad( np.arange( 0, 361) )
    
    plt.polar( theta, spikeFiringRatesFit,label='Firing Rate (spikes/s)')
    plt.legend(loc=8)                   # specify the location of the legend
    plt.title('Polar: ' + title)        # add title to the graph
    

def roll_back( data, degrees ):
    new_xs = data[:,0]
    new_ys = data[:,1]
    
    bin_size = np.abs( data[1,0] - data[0,0] )
    
    old_xs = np.roll( new_xs, -1*degrees )
    old_ys = np.roll( new_ys, -1*degrees )
 
    for i in range(  len( old_xs ) ):
        if( old_xs[i] < 0 ):
            old_xs[i] = old_xs[i-1] + bin_size
          
    return old_xs, old_ys
    
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
    prefered_value = max( fit_curve[:,1] )
    
    preferred_direction = plt.find(  fit_curve[:,1] == prefered_value )
    
    return fit_curve[preferred_direction,0]
    
def fitting( new_xs, new_ys, dir_rates, degrees, title ):
    
    p = fit_tuning_curve( new_xs, new_ys )
    
    curve_xs = np.arange( new_xs[0],new_xs[-1] )
    
    fit_ys = normal_fit( curve_xs,p[0],p[1],p[2] )
   
    fitting_curve = np.column_stack( (curve_xs, fit_ys ) ) 
    
    return fitting_curve
       
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    trials = load_experiment('trials.npy')   
    #spk_times = load_neuraldata('example_spikes.npy')
    spk_times = load_neuraldata('neuron3.npy') 
    
    time_bin = 0.08         # counts the spikes from 100ms before to 100ms after the trial began
    dir_rates,fitting_curve = bin_spikes(trials, spk_times, time_bin)
    
    plot_tuning_curves(dir_rates, 'Example Neuron Tunning Curve')
    plot_fits( dir_rates, fitting_curve, 'Example Neuron Tunning Curve - Fit')


    
    