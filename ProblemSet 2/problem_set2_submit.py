### 
### Hello, curious student!  You shouldn't need to edit anything in this file 
###  (and if you do, there's a good chance you won't be able to upload your
###   answers to the problem set)
###
### All you need to do to submit your assigment is to "run" this file.  
###   In spyder:  load this in the editor and press the green arrow (or F5)
###   From the command prompt: "python problem_set1_submit.py"
###

import urllib
import urllib2
import hashlib
import random
import email
import email.message
import email.encoders
import StringIO
import sys
import datetime

""""""""""""""""""""
""""""""""""""""""""

class NullDevice:
  def write(self, s):
    pass

def submit():   
  print '==\n== [sandbox] Submitting Solutions \n=='
  
  (login, password) = loginPrompt()

  if not login:
    print '!! Submission Cancelled'
    return
  
  print '\n== Connecting to Coursera ... '

  # Part Identifier
#  (partIdx, sid) = partPrompt()

  for partIdx in range(len(partIds)):
    sid = partIds[partIdx]

    print '\n== ' + str(partIdx+1) + ' ' + str(sid)

    # Get Challenge
    (login, ch, state, ch_aux) = getChallenge(login, sid) #sid is the "part identifier"
    if((not login) or (not ch) or (not state)):
      # Some error occured, error string in first return element.
      print '\n!! Error: %s\n' % login
      return
      
    # Attempt Submission with Challenge
    ch_resp = challengeResponse(login, password, ch)
    (result, string) = submitSolution(login, ch_resp, sid, output(partIdx), \
                                        source(partIdx), state, ch_aux)

    print '== %s' % string.strip()


# =========================== LOGIN HELPERS - NO NEED TO CONFIGURE THIS =======================================

def loginPrompt():
  """Prompt the user for login credentials. Returns a tuple (login, password)."""
  (login, password) = basicPrompt()
  return login, password


def basicPrompt():
  """Prompt the user for login credentials. Returns a tuple (login, password)."""
  login = raw_input('Login (Email address): ')
  password = raw_input('One-time Password (from the assignment page. This is NOT your own account\'s password): ')
  return login, password

def partPrompt():
  print 'Hello! These are the assignment parts that you can submit:'
  counter = 0
  for part in partFriendlyNames:
    counter += 1
    print str(counter) + ') ' + partFriendlyNames[counter - 1]
  partIdx = int(raw_input('Please enter which part you want to submit (1-' + str(counter) + '): ')) - 1
  return (partIdx, partIds[partIdx])

def getChallenge(email, sid):
  """Gets the challenge salt from the server. Returns (email,ch,state,ch_aux)."""
  url = challenge_url()
  values = {'email_address' : email, 'assignment_part_sid' : sid, 'response_encoding' : 'delim'}
  data = urllib.urlencode(values)
  req = urllib2.Request(url, data)
  response = urllib2.urlopen(req)
  text = response.read().strip()

  # text is of the form email|ch|signature
  splits = text.split('|')
  if(len(splits) != 9):
    print 'Badly formatted challenge response: %s' % text
    return None
  return (splits[2], splits[4], splits[6], splits[8])

def challengeResponse(email, passwd, challenge):
  sha1 = hashlib.sha1()
  sha1.update("".join([challenge, passwd])) # hash the first elements
  digest = sha1.hexdigest()
  strAnswer = ''
  for i in range(0, len(digest)):
    strAnswer = strAnswer + digest[i]
  return strAnswer 
  
def challenge_url():
  """Returns the challenge url."""
  return "https://class.coursera.org/" + URL + "/assignment/challenge"

def submit_url():
  """Returns the submission url."""
  return "https://class.coursera.org/" + URL + "/assignment/submit"

def submitSolution(email_address, ch_resp, sid, output, source, state, ch_aux):
  """Submits a solution to the server. Returns (result, string)."""
  source_64_msg = email.message.Message()
  source_64_msg.set_payload(source)
  email.encoders.encode_base64(source_64_msg)

  output_64_msg = email.message.Message()
  output_64_msg.set_payload(output)
  email.encoders.encode_base64(output_64_msg)
  values = { 'assignment_part_sid' : sid, \
             'email_address' : email_address, \
             'submission' : output_64_msg.get_payload(), \
             'submission_aux' : source_64_msg.get_payload(), \
             'challenge_response' : ch_resp, \
             'state' : state \
           }
  url = submit_url()  
  data = urllib.urlencode(values)
  req = urllib2.Request(url, data)
  response = urllib2.urlopen(req)
  string = response.read().strip()
  result = 0
  return result, string

## This collects the source code (just for logging purposes) 
def source(partIdx):
  # open the file, get all lines
  f = open(sourceFiles[partIdx])
  src = f.read() 
  f.close()
  return src



############ BEGIN ASSIGNMENT SPECIFIC CODE - YOU'LL HAVE TO EDIT THIS ##############

from problem_set2 import load_experiment, load_neuraldata, bin_spikes
import numpy as np

# Make sure you change this string to the last segment of your class URL.
# For example, if your URL is https://class.coursera.org/pgm-2012-001-staging, set it to "pgm-2012-001-staging".
URL = 'neuraldata-001'

# the "Identifier" you used when creating the part
dev = ""


partIds = ['neuron1-%d%s' % (i,dev) for i in range(1,9)]+['neuron2-%d%s' % (i,dev) for i in range(1,9)]+['neuron3-%d%s' % (i,dev) for i in range(1,9)]
# used to generate readable run-time information for students
partFriendlyNames = ['Neuron1 %d/5' % (i) for i in range(1,9)]+['Neuron2 %d/5' % (i) for i in range(1,9)]+['Neuron3 %d/5' % (i) for i in range(1,9)]
# source files to collect (just for our records)
sourceFiles = ['problem_set2.py']*24

trials = load_experiment('trials.npy')   

          
def order_rates(direction_rates):
    sort_directions = np.argsort(direction_rates[:,0])
    rates = direction_rates[:,1]
    sort_rates = rates[sort_directions]
    return sort_rates
         
def output(partIdx):
  """Uses the student code to compute the output for test cases."""
  outputString = ''

  if partIdx < 8: # This is neuron1
      spikes = load_neuraldata('neuron1.npy')
      direction_rates = bin_spikes(trials,spikes,0.08)
      print( preferred_direction( direction_rates ) )
      hist_vals = order_rates(direction_rates)
      result = hist_vals
      outputString = str(result[partIdx])+'\n'
  elif partIdx < 16:
      spikes = load_neuraldata('neuron2.npy')
      direction_rates = bin_spikes(trials,spikes,0.08)
      print( preferred_direction( direction_rates ) )
      hist_vals = order_rates(direction_rates)
      result = hist_vals
      outputString = str(result[partIdx-8])+'\n'
  else: # This is spike_hard
      spikes = load_neuraldata('neuron3.npy')
      direction_rates = bin_spikes(trials,spikes,0.08)
      print( preferred_direction( direction_rates ) )
      hist_vals = order_rates(direction_rates)
      result = hist_vals
      outputString = str(result[partIdx-16])+'\n'
          
  return outputString.strip()

submit()
