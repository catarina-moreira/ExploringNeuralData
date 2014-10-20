#
#  NAME
#    problem_set1.py
#
#  DESCRIPTION
#    Open, view, and analyze raw extracellular data
#    In Problem Set 1, you will write create and test your own spike detector.
#
#  AUTHOR
#   Catarina Moreira
#   http://web.ist.utl.pt/~catarina.p.moreira/index.html

import numpy as np
import matplotlib.pylab as plt

def load_data(filename):
    """
    load_data takes the file name and reads in the data.  It returns two 
    arrays of data, the first containing the time stamps for when they data
    were recorded (in units of seconds), and the second containing the 
    corresponding voltages recorded (in units of microvolts - uV)
    """
    data = np.load(filename)[()];
    return np.array(data['time']), np.array(data['voltage'])
    
def bad_AP_finder(time,voltage):
    """
    This function takes the following input:
        time - vector where each element is a time in seconds
        voltage - vector where each element is a voltage at a different time
        
        We are assuming that the two vectors are in correspondance (meaning
        that at a given index, the time in one corresponds to the voltage in
        the other). The vectors must be the same size or the code
        won't run
    
    This function returns the following output:
        APTimes - all the times where a spike (action potential) was detected
         
    This function is bad at detecting spikes!!! 
        But it's formated to get you started!
    """
    
    #Let's make sure the input looks at least reasonable
    if (len(voltage) != len(time)):
        print "Can't run - the vectors aren't the same length!"
        APTimes = []
        return APTimes
    
    numAPs = np.random.randint(0,len(time))//10000 #and this is why it's bad!!
 
    # Now just pick 'numAPs' random indices between 0 and len(time)
    APindices = np.random.randint(0,len(time),numAPs)
    
    # By indexing the time array with these indices, we select those times
    APTimes = time[APindices]
    
    # Sort the times
    APTimes = np.sort(APTimes)
    
    return APTimes
    
def good_AP_finder(time,voltage):
    """
    This function takes the following input:
        time - vector where each element is a time in seconds
        voltage - vector where each element is a voltage at a different time
        
        We are assuming that the two vectors are in correspondance (meaning
        that at a given index, the time in one corresponds to the voltage in
        the other). The vectors must be the same size or the code
        won't run
    
    This function returns the following output:
        APTimes - all the times where a spike (action potential) was detected
    """
 
    APTimes = []
       
    #Let's make sure the input looks at least reasonable
    if (len(voltage) != len(time)):
        print "Can't run - the vectors aren't the same length!"
        return APTimes
    
    ##Your Code Here!
    # --------------------------------------------------------------

    if test == 'example':   
        THRESHOLD = 450         # given in homework
        DELAY = 0.0000344       # found by making time[1] - time[0]
        # results for example dataset
        # Correct number of action potentials = 20
        # Percent True Spikes = 100.0
        # False Spike Rate = 0.864891021457 spikes/s
    
    if test == 'easy': 
        THRESHOLD = 250         # found by trial and error
        DELAY = 0.0000222       # found by making time[1] - time[0]
        # results for easy dataset
        # Correct number of action potentials = 10
        # Percent True Spikes = 100.0
        # False Spike Rate = 5.9265857484 spikes/s
    
    if test == 'hard':
        THRESHOLD = 40          # found by trial and error
        DELAY = 0.0000344       # found by making time[1] - time[0]  
        # results for hard dataset
        # Correct number of action potentials = 79
        # Percent True Spikes = 96.2025316456
        # False Spike Rate = 40.3048756544 spikes/s
        
    # Filtering - The first step when processing continuously 
    # recorded data is to apply a band pass filter in order to avoid low 
    # frequency activity and visualize the spikes.
    filtered_data = time[ voltage > THRESHOLD ]    
    
    # add the first data instance to the action potential array
    APTimes.append( filtered_data[0] )       
    
    # filter data: subtract each action potention by its proceeding to see
    # the delay times between them. If the time of the activity is very low,
    # then discard it
    for i in range( len( filtered_data ) - 1 ):  
        if filtered_data[i+1] - filtered_data[i] > DELAY:
            APTimes.append( filtered_data[i] )    
    
    # sorte the array of action potentials and return
    np.sort( APTimes )
    
    # --------------------------------------------------------------
    return APTimes
    

def get_actual_times(dataset):
    """
    Load answers from dataset
    This function takes the following input:
        dataset - name of the dataset to get answers for

    This function returns the following output:
        APTimes - spike times
    """    
    return np.load(dataset)
    
def detector_tester(APTimes, actualTimes):
    """
    returns percentTrueSpikes (% correct detected) and falseSpikeRate
    (extra APs per second of data)
    compares actual spikes times with detected spike times
    This only works if we give you the answers!
    """
    
    JITTER = 0.025 #2 ms of jitter allowed
    
    #first match the two sets of spike times. Anything within JITTER_MS
    #is considered a match (but only one per time frame!)
    
    #order the lists
    detected = np.sort(APTimes)
    actual = np.sort(actualTimes)
    
    #remove spikes with the same times (these are false APs)
    temp = np.append(detected, -1)
    detected = detected[plt.find(plt.diff(temp) != 0)]
 
    #find matching action potentials and mark as matched (trueDetects)
    trueDetects = [];
    for sp in actual:
        z = plt.find((detected >= sp-JITTER) & (detected <= sp+JITTER))
        if len(z)>0:
            for i in z:
                zz = plt.find(trueDetects == detected[i])
                if len(zz) == 0:
                    trueDetects = np.append(trueDetects, detected[i])
                    break;
    percentTrueSpikes = 100.0*len(trueDetects)/len(actualTimes)
    
    #everything else is a false alarm
    totalTime = (actual[len(actual)-1]-actual[0])
    falseSpikeRate = (len(APTimes) - len(actualTimes))/totalTime
    
    print 'Action Potential Detector Performance performance: '
    print '     Correct number of action potentials = ' + str(len(actualTimes))
    print '     Percent True Spikes = ' + str(percentTrueSpikes)
    print '     False Spike Rate = ' + str(falseSpikeRate) + ' spikes/s'
    print 
    return {'Percent True Spikes':percentTrueSpikes, 'False Spike Rate':falseSpikeRate}
    
    
def plot_spikes(time,voltage,APTimes,titlestr):
    """
    plot_spikes takes four arguments - the recording time array, the voltage
    array, the time of the detected action potentials, and the title of your
    plot.  The function creates a labeled plot showing the raw voltage signal
    and indicating the location of detected spikes with red tick marks (|)
    """
    plt.figure()
    
    # Your Code Here
    # --------------------------------------------------------------
    plt.plot(time,voltage, hold=True)       # plot spikes graph
    
    plt.xlabel( 'Time (s)' )                # plot x-axis label
    plt.ylabel( 'Voltage (uV)' )            # plot y-axis label
    plt.title( titlestr )                   # plot graph title

    # specify y tick marks
    y = compute_y_ticks( max(voltage) )
    
    # plot a spike for each action potiential found
    for x in APTimes:
        plt.plot( np.ones( len(y) )*x, y,'k-', color = 'r' )
    
    # --------------------------------------------------------------
    plt.show()

def compute_y_ticks( max_voltage ):

    tick_size = 30    
    y_max_tick = max_voltage + tick_size
    y_min_tick = y_max_tick + tick_size
    
    return [ y_min_tick, y_max_tick ]   
  
def plot_waveforms(time,voltage,APTimes,titlestr):
    """
    plot_waveforms takes four arguments - the recording time array, the voltage
    array, the time of the detected action potentials, and the title of your
    plot.  The function creates a labeled plot showing the waveforms for each
    detected action potential
    """
    
    plt.figure()

    ## Your Code Here 
    # --------------------------------------------------------------

    # from homework examples:
    # action potentials are caught in intervals of 3 miliseconds
    rate = 0.003
    
    # compute the number of action potentials that are measured at each rate
    # this is given by the fraction between the rate and the delayed time between
    # each action potential
    timeStep = time[1]-time[0]
    measurements_per_rate = 2 * int( rate / timeStep )  

    # the x-axis varies between -3 ms to 3 ms and has steps of measurements_per_rate
    time_axis = np.linspace(-rate, rate, measurements_per_rate )  
    voltage_axis = np.zeros( measurements_per_rate )
   
    for val in range(len(APTimes)):  
        
        # find index of action potential (AP) in t data array
        indx = plt.find( time == APTimes[val])
        
        # get data near the action potential  
        if measurements_per_rate / 2 > indx:
            start_indx = 0
        else:
            start_indx = indx - measurements_per_rate / 2
        
        end_indx = indx + measurements_per_rate / 2

        # create a range from the starting index to the ending index
        action_potentials = range(start_indx, end_indx)
    
        # if there are enough points to represent the number of measurements per rate
        if len( action_potentials ) == measurements_per_rate:   
            # then, get the actural voltages from the action potentials identified
            voltage_axis = voltage[ action_potentials ]     
        else:    
            # otherwise, fill the array by adding points to fill up the measurement
            missing = measurements_per_rate - len( action_potentials )
            # for poits very close to zero, fill them with zeros as well
            voltage_axis[ 0 : missing ] = 0          
            # for points not very close to zero, get the actual measurements from action potentials
            voltage_axis[ val:measurements_per_rate ] = voltage[ action_potentials[ 1 ] ] 
                
        # plot the waveform with a blue color      
        plt.plot(time_axis, voltage_axis, color='b', hold=True)

    plt.xlabel( 'Time (s)' )                # plot x-axis label
    plt.ylabel( 'Voltage (uV)' )            # plot y-axis label
    plt.title( titlestr )                   # plot graph title    
          
    plt.show( )                              # show plot
    
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    
    # variable test is a global variable that is used in other functions 
    test = 'example'

    # run the practicing example dataset test
    if test == 'example':   
        t,v = load_data('spikes_example.npy')    
        actualTimes = get_actual_times('spikes_example_answers.npy')
    
    # run the easy dataset test
    if test == 'easy':  
        t,v = load_data('spikes_easy_practice.npy')
        actualTimes = get_actual_times('spikes_easy_practice_answers.npy')
    
    # run the difficult dataset test
    if test == 'hard':
        t,v = load_data('spikes_hard_practice.npy')
        actualTimes = get_actual_times('spikes_hard_practice_answers.npy')
    
    APTime = good_AP_finder( t,v )
    plot_spikes( t, v, APTime, 'Action Potentials from Easy Test Data Set' )
    plot_waveforms( t, v, APTime,'Waveforms' )
    detector_tester( APTime, actualTimes )

