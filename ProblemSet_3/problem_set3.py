import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#from '/home/govind/anaconda/lib/python2.7/site-packages/pandas/' import pandas
#sys.path.append('anaconda/lib/python2.7/site-packages/pandas/')
import pandas as pd

# Note on the data:
# 1. Data contains a set of trials. Each trial is identified by the name of it's target image. (targ)
# 2. Each trial has a unique target image. And each trial starts at time = 0
# 3. When the target image appears, the monkey has to press a button on the left or right.
# 4. During the duration of the trial, eye movements of the monkey are sampled every 5mS. So this is stored as an array.
# 5. For each eye-movement sample, there are three info - (time,hor_position,vert_position)
#On every trial of the experiment, a visual stimulus appeared at a variable time (stimon). (Note that
#each trial starts at its own time zero) The display contained a collection of small images
#displayed on a computer screen. On every trial, the collection included a target image (targ) for
#which the proper response was one of two buttons (side). The target on that trial appeared at the
#coordinates (targ_x, targ_y) which are in a unit called degrees visual angle. Think of this as the
#angle between looking straight ahead and where the target actually appeared. By convention,
#positive values of x are to the right, and positive values of y are up. The actual response (button
#press) occurred at a later time in the trial (response). Eye position was sampled every 5 msec
#(em_time) and the horizontal and vertical positions are separated into separate columns
#(em_horiz, em_vert). Finally, spikes were identified using a spike finder similar in nature to the
#one we used in Problem Set 1. The times for spikes on every trial are in a separate column
#(spk_times).

##
## This function should not be changed
##
def load_data(filename='problem_set3_data.npy'):
    """Read NumPy datafile into python environment and then use pandas to
    convert to a DataFrame to help us manage the data."""
    
    # Read raw data using np.load
    data = np.load(filename)[()]
    
    # Create a DataFrame to organize our data
    df = pd.DataFrame()

    # Add columns we want into the DataFrame
    for column in ['targ', 'em_time', 
                   'em_horiz', 'em_vert', 'stimon', 
                   'response', 'side', 'spk_times']:
        df[column] = data[column]

    # turn targ_pos into separate lists for target x and y position
    df['targ_x'] = np.array([pos[0] for pos in data['targ_pos']])
    df['targ_y'] = np.array([pos[1] for pos in data['targ_pos']])
    return df



##
## This function should be edited as part of Exercise 1
##
def add_info(df):
    """Add additional information to our DataFrame including reaction time
    information and target eccentricity information."""
    ####
    #### Programming Problem 1: Add a columns for computed variables:
    ####                          a. reaction times, called 'rts'
    ####                          b. eccentricity of target, called 'targ_ecc'
    ####

    # a. Reaction Times
    # In the experiment, the stimulus appeared at 'stimon' and the response
    # was recorded at 'response' (both in msec) 

    # Uncomment the following line and add the df['rts'] column 
    #   Hint: refer to df['stimon'] and df['response'] !!!

    # df['rts'] = *** YOUR CODE HERE ***


    # b. Target Eccentricity
    # The target x,y position is specified by df['targ_x'] and df['targ_y']
    #
    # Here we want to turn this into a column that specifies the distance 
    # of the target from 0,0 (the center of the screen):
    #
    #   Hints: You will need the pythagorean theorem: ecc = sqrt(x^2+y^2) 
    #          To square an array, you can multiply it by itself or do array**2
    #          You can use np.sum (or +), np.sqrt, and np.round here

    # df['targ_ecc'] = *** YOUR CODE_HERE ***

    df['rts'] = np.abs(df['stimon'] - df['response']) # New column for response time
    print len(df['rts'])
    df['targ_ecc'] = np.round(np.sqrt(df['targ_x']*df['targ_x'] + df['targ_y']*df['targ_y']),5) # New column for eccentricity of the target
    #print np.unique(df['targ_ecc']) # See the unique number of target eccentricities tested
    #print df.groupby('stimon').first()
    print df['stimon'].head(10)
    print df[df['stimon'] <=0]
    is_null = pd.isnull(df['stimon'] )
    
    for i in range(0,len(is_null)):
        if(is_null[i] == True):
            print "Null at index" + str(i)
            
    #print df['targ_y'].head(10)
    print df['targ_ecc'].value_counts() # see the number of trials for each eccentricities in the data set
    print df['targ'].value_counts() # See the number of trials in the data set for each trial type
    
    # Leave this here - it adds information used for Exercise 5
    add_acq_time(df)
    
    return df

##
## This function should be edited as part of Exercise 2
##
def rts_by_targ_ecc(df):
    """Use pandas to compute the mean of all the reaction times,
    sorted by the targ_ecc variable (how far the target was from the
    center of the screen).  Returns a pandas series."""

    ####
    #### Programming Problem 2: 
    ####        Use the pandas "pivot_table" command to summarize data
    ####

    # results = df.pivot_table( *** YOUR CODE HERE *** )
    results = df.pivot_table(values='rts', index='targ_ecc', aggfunc=np.mean)
    return results


##
## This function should be edited as part of Exercise 3
##
def plot_rts(df):
    """Use pandas to compute the mean of all the reaction times,
    sorted by the targ_ecc variable and side and then plot the results."""

    ####
    #### Programming Problem 3: Use the pandas to create barchart of sorted rts
    ####

    # df['side_name'] = ### YOUR CODE HERE
    # replace 0 in side column with left and 1 with right
    df['side_name'] = np.choose(df['side'],['left','right'])

    # Now create a pivot table (specifying values, index, and columns)
    results = df.pivot_table(values='rts', index='targ_ecc', columns='side_name',aggfunc=np.mean)
    # And now plot that pivot table
    results.plot(kind='bar')
    # And add plot details (title, legend, xlabel, and ylabel)
    plt.title("Reaction time for each target eccentricity")
    plt.xlabel("Reaction time in mS")
    plt.ylabel("Target Eccentricities")
##
## This function should be edited as part of Exercise 4
##
def get_ems(df, trial):
    """Extract the eye movement times, horizontal, and vertical positions for
    a given trial, selecting only those times when the stimulus is visible
    (between stimon and response).  Returns times, horizontal, and vertical
    arrays."""

    ####
    #### Programming Problem 4: 
    ####     Extract eye movement data for a single trial here
    ####

    #### Hints
    ####  Remember that to get information about a given trial, you
    ####  will need to use that trial as part of the index.  For example,
    ####  to get the time the stimulus appeared on trial 14, you would write:
    ####    df['stimon'][14]

    # t = *** YOUR CODE HERE ***
    # h = *** YOUR CODE HERE ***
    # v = *** YOUR CODE HERE ***
    # Extract the start time for this trial
    SR = df['em_time'][0][1] - df['em_time'][0][0]
    t1 = df['stimon'][trial] - df['stimon'][trial] % SR 
    t2 = df['response'][trial] + df['response'][trial] % SR
    emt = df['em_time'][trial]
   
    # FInd the total number of samples in range, so that we can preallocate arrays for t,h,v
   # and get the intex of each in range value
   # There should be a simpler way to do this in Python, but I dont know it :(
    cnt = 0
    indices = np.zeros(len(emt))
    for i in range(0,len(emt)):
        if(emt[i] >= t1 and emt[i] <= t2):
            indices[cnt] = i
            cnt = cnt + 1

    print "Time stamps in range = " + str(cnt)
    t = np.zeros(cnt)
    h = np.zeros(cnt)
    v = np.zeros(cnt)
    
    for i in range(0,cnt):
        t[i] = emt[indices[i]]
        h[i] = df['em_horiz'][trial][indices[i]]
        v[i] = df['em_vert'][trial][indices[i]]


    return t, h, v

##
## This function should be edited as part of Exercise 4
##
def plot_ems_and_target(df, trial):
    """Plot the eye movement traces for the horizontal and vertical eye
    positions along with two horizontal lines showing the target position
    for a given trial."""

    ####
    #### Programming Problem 5: 
    ####    Plot eye movements and target location on a single plot
    ####

    t, h, v = get_ems(df, trial)
    # Can get information about target location here

    plt.figure()

    # *** YOUR CODE HERE ***
    #horizontal lines for target position
    th = df['targ_x'][trial] * np.ones(len(t))
    tv = df['targ_y'][trial] * np.ones(len(t))
    plt.plot(t, h, 'r', t, v, 'g',t,th,'r', t,tv,'g')
    plt.ylim(-10,10)
    plt.title("Eye Movements for Trial " + str(trial))
    plt.xlabel("Time in mS")
    plt.ylabel("Position in degrees visual angle")
    plt.show()
    


##
## This function should be edited as part of Exercise 5
##

def get_rate(spk_times, start, stop):
    """Return the rate for a single set of spike times given 
    a spike counting interval of start to stop (inclusive)."""

    ####
    #### Programming Problem 6: 
    ####    Get rate from list of spk_times and [start,stop) window
    ####

    # rate = *** YOUR CODE HERE ***
    # Remember that rate should be in the units spikes/sec
    # but start and stop are in msec (.001 sec)
    
    # To find the rate just count the number of spikes within the start and stop time and divide by time in seconds
    dt = (stop - start)
    #rate = ((start <= spk_times) & (spk_times <= stop)).sum()/dt
    rate = np.count_nonzero((start < spk_times) & (spk_times <= stop)) * 1000/dt
    
    print "Firing Rate = " + str(rate)

    return rate


##
## This function should not need to be edited
##
def add_aligned_rates(df, alignto, start, stop):
    """Use the get_rate() function to add rates to a DataFrame where the
    counting window is [alignto_event+start, alignto_event+stop).  If, for
    example, alignto='stimon', then the windows is [stimon+start,stimon+stop).
    Nothing is returned, but the DataFrame has a new column added.  E.g., 
    add_aligned_rates(df, 'stimon', 100, 200)
    will add a new column to df called df['rates_stimon_100_200']
    ."""
    
    spks = df['spk_times']
    align = df[alignto]
    rates = [get_rate(spks[i],align[i]+start,align[i]+stop) 
             for i in range(len(df))]
    df['rates_'+alignto+'_'+str(start)+'_'+str(stop)] = np.array(rates)


### NO NEED TO EDIT BELOW HERE (examine, if you wish!)

#
#  Code for finding the time the target was "looked at" (acquired)
#    DO NOT EDIT, as this will affect your problem set, but you are
#    welcome to see one way that we can find the time that the eye position
#    gets within a certain distance of the target.  This code is a little
#    tricky, because if the eye looks past the target during an eye movement
#    we don't want to count that.  Only new "fixations" near the target are
#    counted.
#

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def find_targ_acquired_time(d,threshold,stimon,runsize=4,sample_period=5):

    """Finds time when distance values in d array are below threshold for
    at least runsize values in a row.  Returns time (in ms)."""

    regions = contiguous_regions(np.less(d,threshold))
    longruns = np.greater(regions[:,1]-regions[:,0],runsize)
    after_stimon = np.greater(np.multiply(regions[:,1],sample_period),stimon)
    longruns_after_stimon = np.nonzero(np.logical_and(longruns,after_stimon))
    return regions[longruns_after_stimon][0][0]*sample_period

def add_acq_time(df):
    h_dist = df['em_horiz']-df['targ_x']
    v_dist = df['em_vert']-df['targ_y']
    ss = h_dist*h_dist+v_dist*v_dist
    d = [np.sqrt(eyedist) for eyedist in ss]
    n = len(d)
    df['targ_acq'] = [find_targ_acquired_time(d[i],1.5, df['stimon'][i])
                      for i in range(n)]


# Code to run for testing if this module is run directly
if __name__ == "__main__":
    df = load_data()
    df2 = add_info(df)
    # Find the mean reaction time for each of the eccentricities in the table
    print df2.pivot_table(values='rts', index='targ_ecc', aggfunc=np.mean)
    #plot_rts(df2)
    #t,h,v = get_ems(df, 0)
    #plot_ems_and_target(df2,213)
    #print( get_rate(np.arange(1000,step=10),100,200) ) 
    add_aligned_rates(df2, 'stimon' ,100, 200)
    df.pivot_table(values='rates_stimon_100_200', index='targ',columns='targ_ecc')
    add_aligned_rates(df2, 'targ_acq' ,100, 200)
    print(df.pivot_table(values='rates_targ_acq_100_200', index='targ',columns='targ_ecc'))