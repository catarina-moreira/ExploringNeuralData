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
from random import randint

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
    #spikes_per_trial = count_spikes_per_trial( spk_times, trials, time_bin  )
    
    # Group the trials by direction of motion & compute the average of the trials and spiker per motion
    dir_rates = compute_firing_rate_per_motion( direction_of_motions, spk_times, trials, time_bin )
        
    return dir_rates

# Group the trials by direction of motion & compute the average of the trials and spiker per motion
def compute_firing_rate_per_motion( direction_of_motions, spk_times, trials, time_bin ):
    
   avg_firing_rate = np.zeros( len( direction_of_motions ) )
   
   for d in range( len( direction_of_motions ) ):
       
       # extract the trial times for each direction
       time_indx = plt.find( direction_of_motions[d] == trials[:,0] ) 
       times_motion_started = trials[ time_indx, 1 ]
       
       total_spikes_per_trial = 0
       
       # for each time of the trial where the movement started,
       # count the number of spikes that occurred in the time window
       for t in range( len(times_motion_started) ):
           spike_indx = plt.find( (spk_times >= times_motion_started[t] - time_bin) & (spk_times <= times_motion_started[t] + time_bin)  )
           total_spikes_per_trial += len( spike_indx )
       
       # compute the average firing rate
       avg_rate = ( (1.0*total_spikes_per_trial)  / (1.0*len(times_motion_started))) / time_bin
       avg_rate /= 2
       
       # append the firing rate to 
       avg_firing_rate[d] = avg_rate
          
   #append the direction d and the firing rate to the initially empty array
   dir_rates = np.column_stack( (direction_of_motions, avg_firing_rate)  )
   
   return dir_rates
    
   
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
    plt.bar( direction_rates[:,0], direction_rates[:,1],  width = bin_size, align='center' )
    plt.xticks( np.arange(0, 361, bin_size) )       # specify the range of ticks
    
    #plt.xlim( ( min_x,  max_x ) )
    #plt.ylim( ( min_y,  max_y) )
    plt.xlabel('Direction of Motion (degrees)')     # add label to x-axis
    plt.ylabel('Firing Rate (spikes/s)')          # add label to y-axis
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
    #plt.legend(loc=8)                   # specify the location of the legend
    plt.title('Polar: ' + title)        # add title to the graph
    
def roll_axes( direction_rates ):
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
    
    #debug_plot_rolled_data( new_xs, new_ys, bin_size  )
     
    return new_xs, new_ys, int( roll_degrees*bin_size);
    
def debug_plot_rolled_data( new_xs, new_ys, bin_size  ):
    
    plt.bar( new_xs, new_ys, width=45, align='center')
    plt.xticks( np.arange( np.min(new_xs) ,  np.max(new_xs)+bin_size , bin_size  ))
    plt.xlim( ( np.min(new_xs) - bin_size ,  np.max(new_xs) + bin_size)  )
    
def normal_fit(x,mu, sigma, A):
    """
    This creates a normal curve over the values in x with mean mu and
    variance sigma.  It is scaled up to height A.
    """
    n = A*mlab.normpdf(x,mu,sigma)
    return n

def fit_tuning_curve_normal(centered_x, centered_y):
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
    
def fit_tuning_curve_von_mises(centered_x, centered_y):
    """
    This takes our rolled curve, generates the guesses for the fit function,
    and runs the fit.  It returns the parameters to generate the curve.
    """

    max_y = np.amax( centered_y )                       # What is the biggest y-value? (This
                                                        # estimates the amplitude of the curve
    
    max_x = centered_x[ np.argmax( centered_y) ]        # Where is the biggest y-value? (This 
                                                        # estimates the mean of the curve, mu)
    
    p, cov = optimize.curve_fit(von_mises_fitfunc, centered_x, centered_y, p0=[ max_y, 4, max_x, 1 ])

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
    plt.ylabel('Firing Rate (spikes/s)')
    plt.title( title )
    
def plot_polar_normal_fit( direction_rates, fit_curve, title ):
    
    bin_size = 45
    
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

def plot_polar_von_mises_fit( direction_rates, fit_curve, title ):
    
    bin_size = 45
    
    spikeFiringRates = direction_rates[:,1]  # select the spikes firing rates     

    # the polar representation is like a circle, so we correct the data
    # in a way that the last element of the array if the same as the first one
    spikeFiringRates = np.append(spikeFiringRates, spikeFiringRates[0])
    
    print( spikeFiringRates )
    # compute the angles . A circle ranges from 0 to 360 deg
    theta = np.append( direction_rates[:,0], np.pi*2+0.01 )
   
    print( len( theta ) )
    print( len(spikeFiringRates) )
    
    plt.subplot(2,2,4,polar=True)
    # plot the polar graph
    plt.polar( theta, spikeFiringRates, 'o' )
    
    spikeFiringRatesFit = fit_curve[:,1]
    spikeFiringRatesFit = np.append(spikeFiringRatesFit, spikeFiringRatesFit[0])
    
    
    theta = np.arange( 0, 2*np.pi, 0.01) 
    
    plt.polar( theta, spikeFiringRatesFit,label='Firing Rate (spikes/s)')
    plt.legend(loc=8)                   # specify the location of the legend
    plt.title('Polar: ' + title)        # add title to the graph
    
def roll_back( new_xs, new_ys, degrees ):

    # compute bin_size
    bin_size = np.abs( new_xs[1] - new_xs[0] )
    
    # roll the data in the oposite direction 
    old_xs = np.roll( new_xs, int(-1*degrees) )
    old_ys = np.roll( new_ys, int(-1*degrees) )
 
    # rename the entries of the x-axis
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
    
    # find the motion that is associated with the highest firing rate
    prefered_value = max( fit_curve[:,1] )
    preferred_direction = plt.find(  fit_curve[:,1] == prefered_value )
    
    return fit_curve[preferred_direction[0],0]
    
def fitting_normal( dir_rates ):
    
    # start the fitting process
    # we need to fit the data in a normal distribution
    # in order todo so, we need to change the direction rates in such a way,
    # that the maximum peak of the function is aligned with the center of the 
    # histogram

    # we will perform a shift on the data in order to put the maximum peak in 
    # the middle of the histogram, just like in a normal distribution
    new_xs, new_ys, degrees = roll_axes( dir_rates )
    
    # next, we will try to fit the normal distribution in the data. 
    # learn a function that fits the data
    p = fit_tuning_curve_normal( new_xs, new_ys )
    
    # improve the curve by adding more data in the x-axis (more degrees of motion)
    curve_xs = np.arange( new_xs[0],new_xs[-1] )
    
    # apply the learning function previously computed to the new range of motions
    fit_ys = normal_fit( curve_xs, p[0], p[1], p[2] )
   
    # shift the data back
    new_xs, new_ys = roll_back( curve_xs, fit_ys, degrees )       
    
    # combine the results and return
    fitting_curve = np.column_stack( (new_xs, new_ys ) ) 
    
    return fitting_curve

def fitting_von_mises( dir_rates ):
    
    # start the fitting process
    # we need to fit the data in a normal distribution
    # in order todo so, we need to change the direction rates in such a way,
    # that the maximum peak of the function is aligned with the center of the 
    # histogram

    # we will perform a shift on the data in order to put the maximum peak in 
    # the middle of the histogram, just like in a normal distribution
    new_xs, new_ys, degrees = roll_axes( dir_rates )
    
    # next, we will try to fit the normal distribution in the data
    p = fit_tuning_curve_von_mises( new_xs, new_ys )
   
    # improve the curve by adding more data in the x-axis (more degrees of motion)
    curve_xs = np.deg2rad( np.arange(new_xs[0], new_xs[-1]  ) )
    curve_xs = np.deg2rad( np.arange(new_xs[0], new_xs[-1], 0.01 ))
    
    # apply the learning function previously computed to the new range of motions
    fit_ys = von_mises_fitfunc( curve_xs, p[0], p[1], p[2], p[3] )
    
    # shift the data back
    new_xs, new_ys = roll_back( curve_xs, fit_ys, degrees )      
    
    # combine the results and return
    fitting_curve = np.column_stack( (new_xs, new_ys ) ) 
    
    return fitting_curve

##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    trials = load_experiment('trials.npy')   
    spk_times = load_neuraldata('example_spikes.npy')
    #spk_times = load_neuraldata('neuron1.npy') 
    
    time_bin = 0.1      # counts the spikes from 100ms before to 100ms after the trial began
    dir_rates = bin_spikes(trials, spk_times, time_bin)
    
    #fitting_curve = fitting_normal( dir_rates )
    fitting_curve = fitting_normal( dir_rates )    
    
    plot_tuning_curves(dir_rates, 'Neuron Tunning Curve')
    
    plot_fits( dir_rates, fitting_curve, 'Neuron Tunning Curve - Fitting')
    
    plot_polar_normal_fit( dir_rates, fitting_curve, 'Neuron Tunning Curve - Fitting' )
    # Homework Question 1
    # 136
    # print( len(trials) )
    
    # Homework Question 2
    # 17
    # indx = plt.find( trials[:,0] == 45  )
    # print( len( trials[indx,1] ) )
    
    # Homework Question 6
    # 132.0
    #pd = preferred_direction( fitting_curve )    
    #print( pd )

    # Homework Question 7
    # 143.0
    
    # Homework Question 8
    # 259.0 ( wrong...)
    
    