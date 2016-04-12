from scipy.io import loadmat
import numpy as np

############### class definitions #################



class runParams:
    def __init__(self,money,orderImages,scannerKey,responseKeys,seed,readyTime,ti,stimuliTime,choiceTime,feedbackTime,fixationTime):
        self.money=money
        self.orderImages=orderImages
        self.scannerKey=scannerKey
        self.responseKeys=responseKeys
        self.seed=seed
        self.readyTime=readyTime
        self.ti=ti
        self.stimuliTime=stimuliTime
        self.choiceTime=choiceTime
        self.feedbackTime=feedbackTime
        self.fixationTime=fixationTime

class session:
    def __init__(self, trial, trialType,timeStimPresn,choice,response,feedback,rt,hesit,runParams):
        self.trial=trial
        self.trialType=trialType
        self.timeStimPresn=timeStimPresn
        self.choice=choice
        self.response=response
        self.feedback=feedback
        self.rt=rt
        self.hesit=hesit
        self.runParams=runParams #get these from the above class


class ppt:
    'the class associated with an individual participant'
    def __init__(self, studyNumber, studyArm,sessions):
        self.studyNumber = studyNumber
        self.studyArm   = studyArm
        self.sessions = sessions

#some method to add in sessions to the ppt class.

#some method to pull out only the look loss gain trials

#some method to average these over certain ppts


# some class that defines a study?

###################################################

############## function definitions ###############

def readReturnSession(filename,session):
    'give .mat file from the RLtask and session number associated with the ppt, returns python object'
    matStruct = loadmat(filename)
    theseRunParams=runParams(matStruct['money'][0],matStruct['orderImages'][0],matStruct['SCANNERKEY'][0],
                             matStruct['RESPONSEKEYS'][0],matStruct['seed'][0],matStruct['READYTIME'][0],
                             matStruct['ti'][0],matStruct['STIMULITIME'][0],matStruct['CHOICETIME'][0],
                             matStruct['FEEDBACKTIME'][0],matStruct['FIXATIONTIME'][0])
                             
    matStructData=matStruct['data']
    
    theseData=session(session,matStructData[:,1],matStructData[:,2],matStructData[:,3],
                      matStructData[:,4],matStructData[:,5],matStructData[:,6],
                      matStructData[:,7],matStructData[:,8], theseRunParams)

    return theseData

###################################################

################## dictionaries ###################

#ppt number, study arm
studyArmDict={
                100:1,
            }

###################################################



#.txt file with all the .mat names that are being analysed and bin, has filename and the bin
# study arms:
# 1: control no treatment
# 2: HAM-D < 7 on medication
# 3: HAM-D > 13 on medication
# 4: HAM-D > 17 no medication
matfiles="/Users/cc401/data/BIODEP/RLtask/testfilelist.txt"

anadir="/Users/cc401/data/BIODEP/RLtask/testing/" #analysis output directory


#read in all the text files in the file list
fileList = np.loadtxt(matfiles,unpack=True,dtype='string')

for filename in fileList:
    pptNum
    #renaming session numbers to just be 1, 2
    aSession=readReturnSession(filename,session)





#create a list of ppts