#
#  NAME
#    problem_set4.py
#
#  DESCRIPTION
#    In Problem Set 4, you will classify EEG data into NREM sleep stages and
#    create spectrograms and hypnograms.
#

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as m
import os,sys, getopt

def load_examples(filename):
    """
    load_examples takes the file name and reads in the data.  It returns an
    array containing the 4 examples of the 4 stages in its rows (row 0 = REM;
    1 = stage 1 NREM; 2 = stage 2; 3 = stage 3 and 4) and the sampling rate for
    the data in Hz (samples per second).
    """
    data = np.load(filename)
    return data['examples'], int(data['srate'])

def load_eeg(filename):
    """
    load_eeg takes the file name and reads in the data.  It returns an
    array containing EEG data and the sampling rate for
    the data in Hz (samples per second).
    """
    data = np.load(filename)
    return data['eeg'], int(data['srate'])

def load_stages(filename):
    """
    load_stages takes the file name and reads in the stages data.  It returns an
    array containing the correct stages (one for each 30s epoch)
    """
    data = np.load(filename)
    return data['stages']

def plot_example_psds(example,rate):
    """
    This function creates a figure with 4 lines to show the overall psd for 
    the four sleep examples. (Recall row 0 is REM, rows 1-3 are NREM stages 1,
    2 and 3/4)
        
    """

    sleep_stages = ['REM sleep', 'Stage 1 NREM sleep', 'Stage 2 NREM sleep', 'Stage 3 and 4 NREM sleep'];    
    
    plt.figure()
    
    ##YOUR CODE HERE    
    for i in range( len( example[:,0]) ):    
        
        # Apply power spectral density using a Fast Fourier Transform 
        # to generate blocks of data
        psd, frequency = m.psd(example[i, :], NFFT = 512, Fs = rate )
        
        # normalize frequency
        psd = psd / np.sum(psd)
        
        # plot sleep stages
        plt.plot(frequency, psd, label = sleep_stages[i])
        
        # add legend
        plt.ylabel('Normalized Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
    
    plt.xlim(0,20)
    plt.legend(loc=0)
    plt.title('Overall ower Spectral Density for Sleep Stages')
    
    return

def plot_example_spectrograms(example,rate):
    """
    This function creates a figure with spectrogram sublpots to of the four
    sleep examples. (Recall row 0 is REM, rows 1-3 are NREM stages 1,
    2 and 3/4)
    """
    sleep_stages = ['REM sleep', 'Stage 1 NREM sleep', 'Stage 2 NREM sleep', 'Stage 3 and 4 NREM sleep'];    
    
    plt.figure()
    
    ###YOUR CODE HERE
    for i in range( len(example[:,0]) ):
       
       # plot every sleep stage in a separate plot
       plt.subplot(2,2,i+1)
       
       # plot spectogram
       plt.specgram(example[i, :],NFFT=512,Fs=rate)   
       
       # add legend
       plt.xlabel('Time (Seconds)')
       plt.ylabel('Frequency (Hz)')
       plt.title( 'Spectogram ' + sleep_stages[i] )
       
       plt.ylim(0,60)
       plt.xlim(0,290)
    return
      
            
def classify_epoch(epoch,rate):
   """
   This function returns a sleep stage classification (integers: 1 for NREM
   stage 1, 2 for NREM stage 2, and 3 for NREM stage 3/4) given an epoch of 
   EEG and a sampling rate.
   """
   
   ###YOUR CODE HERE
   stage = 2
   
   if( np.var(epoch) > 2000  ):
       return 3
   
   if( np.var(epoch) < 250 ):
       return 1
   
   if( np.max(epoch) < 50  ):
       return 1
   
   
       
   #if( np.min(epoch) < -35 + search_ball  ):
   #   return 1
      
   
   
   return stage
   

def plot_hypnogram(eeg, stages, srate):
    """
    This function takes the eeg, the stages and sampling rate and draws a 
    hypnogram over the spectrogram of the data.
    """
    
    fig,ax1 = plt.subplots()  #Needed for the multiple y-axes
    
    #Use the specgram function to draw the spectrogram as usual
    psd, frequency, bins, im = plt.specgram(eeg, NFFT=512, Fs=srate) 
    
    #Label your x and y axes and set the y limits for the spectrogram
    plt.ylim(0,30)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Frequency (Hz)')
    
    ax2 = ax1.twinx() #Necessary for multiple y-axes
    
    #Use ax2.plot to draw the hypnogram.  Be sure your x values are in seconds
    #HINT: Use drawstyle='steps' to allow step functions in your plot
    
    times = np.arange(0,len(stages)*30, 30)  
    ax2.plot(times, stages, drawstyle='steps', linewidth = 2)
    
    #Label your right y-axis and change the text color to match your plot
    ax2.set_ylabel('NREM Stage',color='b')

    #Set the limits for the y-axis 
    plt.ylim(0.5,3.5)
    
    #Set the limits for the y-axis 
    plt.xlim(0,3000)
    
    #Only display the possible values for the stages
    ax2.set_yticks(np.arange(1,4))
    
    #Change the left axis tick color to match your plot
    for t1 in ax2.get_yticklabels():
        t1.set_color('b')
        
    
    #Title your plot    
    plt.title('Hypnogram - Test Data') 

        
def classifier_tester(classifiedEEG, actualEEG):
    """
    returns percent of 30s epochs correctly classified
    """
    epochs = len(classifiedEEG)
    incorrect = np.nonzero(classifiedEEG-actualEEG)[0]
    percorrect = (epochs - len(incorrect))/epochs*100
    
    for i in range( epochs ):
        if( classifiedEEG[i] != actualEEG[i] ):
           print 'actualEEG ' + str(actualEEG[i])+ ' classifiedEEG ' + str(classifiedEEG[i])
           print('---------------------------')
            
    print 'EEG Classifier Performance: '
    print '     Correct Epochs = ' + str(epochs-len(incorrect))
    print '     Incorrect Epochs = ' + str(len(incorrect))
    print '     Percent Correct= ' + str(percorrect) 
    print 
    return percorrect
  
    
def test_examples(examples, srate):
    """
    This is one example of how you might write the code to test the provided 
    examples.
    """
    i = 0
    bin_size = 30*srate
    c = np.zeros((4,len(examples[1,:])/bin_size))
    while i + bin_size < len(examples[1,:]):
        for j in range(1,4):
            c[j,i/bin_size] = classify_epoch(examples[j,range(i,i+bin_size)],srate)
        i = i + bin_size
    
    totalcorrect = 0
    num_examples = 0
    for j in range(1,4):
        canswers = np.ones(len(c[j,:]))*j
        print(canswers)
        correct = classifier_tester(c[j,:],canswers)
        totalcorrect = totalcorrect + correct
        num_examples = num_examples + 1
    
    average_percent_correct = totalcorrect/num_examples
    print 'Average Percent Correct= ' + str(average_percent_correct) 
    return average_percent_correct

def classify_eeg(eeg,srate):
    """
    DO NOT MODIFY THIS FUNCTION
    classify_eeg takes an array of eeg amplitude values and a sampling rate and 
    breaks it into 30s epochs for classification with the classify_epoch function.
    It returns an array of the classified stages.
    """
    bin_size_sec = 30
    bin_size_samp = bin_size_sec*srate
    t = 0
    classified = np.zeros(len(eeg)/bin_size_samp)
    while t + bin_size_samp < len(eeg):
       classified[t/bin_size_samp] = classify_epoch(eeg[range(t,t+bin_size_samp)],srate)
       t = t + bin_size_samp
    return classified
        
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    #YOUR CODE HERE

    plt.close('all') #Closes old plots.
    
    ##PART 1
    #Load the example data
    examples, srate = load_examples('example_stages.npz')
    
    #Plot the psds
    plot_example_psds(examples,srate)
    
    #Plot the spectrograms
    plot_example_spectrograms(examples,srate)
    
    test_examples(examples, srate)

    #Load the practice data
    eeg, srate = load_eeg('practice_eeg.npz')   
    #Load the practice answers
    stages = load_stages('practice_answers.npz')
    
    #Classify the practice data
    classifiedEEG = classify_eeg(eeg,srate)  
    
    #Check your performance
    actualEEG = stages
    classifier_tester(classifiedEEG, actualEEG)
    
    #Generate the hypnogram plots
    plot_hypnogram(eeg, classifiedEEG, srate)

     #Load the practice data
    eeg, srate = load_eeg('test_eeg.npz')   
    
    #Classify the practice data
    classifiedEEG = classify_eeg(eeg,srate)  
    
    #Generate the hypnogram plots
    plot_hypnogram(eeg, classifiedEEG, srate)



    

    
