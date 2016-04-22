from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import glob

############### class definitions #################


class RunParams:
    '''A set of run parameters that can be attached to a Session object'''
    def __init__(self,money,orderImages,scannerKey,responseKeys,seed,readyTime,ti,stimuliTime,choiceTime,feedbackTime,fixationTime,TR=None):
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
        self.TR = TR #can specify in a function call or not or can access later. This is not by standard recorded in the .mat file.


class Session:
    '''class associated with an individual session'''
    def __init__(self, sessionName, trial, trialType,timeStimPresn,choice,response,feedback,rt,hesit,runParams):
        self.sessionName=sessionName
        self.trial=trial
        self.trialType=trialType
        self.timeStimPresn=timeStimPresn
        self.choice=choice
        self.response=response
        self.feedback=feedback
        self.rt=rt
        self.hesit=hesit
        self.runParams=runParams #get these from the above class
        self.alphas={'gain':0, 'look':0, 'loss':0}
        self.betas={'gain':0, 'look':0, 'loss':0}
        self.rewardWeights={'gain':0, 'look':0, 'loss':0}
    
    def getGainTrials(self):
        '''get indices of gain trials in sessions'''
        return np.where(self.trialType == 0)[0]
    
    def getLookTrials(self):
        '''get indices of look trials in sessions.'''
        return np.where(self.trialType == 1)[0]
    
    def getLossTrials(self):
        '''get indices of loss trials in sessions'''
        return np.where(self.trialType == 2)[0]




class Ppt:
    '''the class associated with an individual participant'''
    def __init__(self, pptName, studyArm, sessions=None):
        self.pptName = pptName
        self.studyArm   = studyArm
        
        self.sessions = {}
        try:
            sessNames=[sessions[i].sessionName for i in len(sessions)]
            self.sessions.update(dict(zip(sessNames, sessions)))
        except:
            self.sessions.update({sessions.sessionName : sessions})


    def __setitem__(self, key, item):
        self.sessions[key]=item
    
    def __getitem__(self, key):
        return self.sessions[key]
    
    def listSessions(self):
        '''returns a list of sessions attached to the participant'''
        return self.sessions.keys()
    
    def addSession(self, session):
        '''add a session to the ppt'''
        self.sessions.__setitem__(session.sessionName,session)



# some class that defines a study?
class Study:
    ''' define a study with a name and a list of ppts'''
    def __init__(self, studyName, ppts=None):
        '''declare a new study with participants'''
        self.studyName=studyName
        self.ppts={}
        if ppts is not None:
            try:
                pptNames=[ppts[i].pptName for i in len(ppts)]
                self.ppts.update(dict(zip(pptNames, ppts)))
            except:
                self.ppts.update({ppts.pptName : ppts})




    def __setitem__(self, key, item):
        '''add a new session'''
        self.ppts[key]=item
    
    def __getitem__(self, key):
        '''return a session'''
        return self.ppts[key]
    
    def listPpts(self):
        '''returns a list of ppts attached to the study'''
        return self.ppts.keys()
    
    def addPpt(ppt):
        '''adds an existing participant to a study'''
        self.ppts.__setitem__(ppt.pptName, ppt)
    
    def addNewPpt(self, pptName, studyArm, aSession=None):
        '''makes a new participant and adds to a study'''
        orphanPpt=Ppt(pptName, studyArm, aSession)
        self.ppts.__setitem__(orphanPpt.pptName, orphanPpt)
    
    def printResults(self):
        '''prints all alpha, beta, reward weight numbers from the study for each participant and session'''
        print "======================================================="
        print "RESULTS FOR STUDY: "+self.studyName
        print "======================================================="
        print "-------------------------------------------------------"
        for i, ppt in enumerate(self.listPpts()):
            thisPpt=self.ppts[ppt]
            print "-PPT: "+ppt
            for j, session in enumerate(self.ppts[ppt].listSessions()):
                thisSession=thisPpt.sessions[session]
                print "--SESSION: "+session
                print "---GAIN: alpha: "+str(thisSession.alphas['gain']) \
                    +"\tbeta: "+str(thisSession.betas['gain']) \
                        +"\treward weighting: " +str(thisSession.rewardWeights['gain'])
                #Sometimes multiple solutions will be outputted, this tags the result output explaining
                if len(thisSession.alphas['gain']) > 1: print "MULTIPLE GAIN SOLUTIONS DETECTED"
                print "---LOOK: alpha: "+str(thisSession.alphas['look']) \
                    +"\tbeta: "+str(thisSession.betas['look']) \
                        +"\treward weighting: "+str(thisSession.rewardWeights['look'])
                if len(thisSession.alphas['look']) > 1: print "MULTIPLE LOOK SOLUTIONS DETECTED"
                print "---LOSS: alpha: "+str(thisSession.alphas['loss']) \
                    +"\tbeta: "+str(thisSession.betas['loss']) \
                        +"\treward weighting: "+str(thisSession.rewardWeights['loss'])
                if len(thisSession.alphas['loss']) > 1: print "MULTIPLE LOSS SOLUTIONS DETECTED"
                print "-------------------------------------------------------"
                print "-------------------------------------------------------"


###################################################

############## function definitions ###############
##
def readReturnSession(filename):
    '''
        give .mat file from the RLtask and session number associated with the ppt, returns python object.
        NB trial starts at 1 in matlab but changed to start at trial 0 in python for consistency
        with python indexing.
        '''
    matStruct = loadmat(filename)
    theseRunParams=RunParams(matStruct['money'][0],matStruct['orderImages'][0],matStruct['SCANNERKEY'][0],
                             matStruct['RESPONSEKEYS'][0],matStruct['seed'][0],matStruct['READYTIME'][0],
                             matStruct['ti'][0],matStruct['STIMULITIME'][0],matStruct['CHOICETIME'][0],
                             matStruct['FEEDBACKTIME'][0],matStruct['FIXATIONTIME'][0])

    sessionName=filename.split('/')[-1].split('.')[0].split('_')[-1][4:]
                             
    matStructData=matStruct['data']
                             
    #NB, trial indexing starts at zero
    theseData=Session(sessionName,matStructData[:,1]-1,matStructData[:,2],matStructData[:,3],
                                  matStructData[:,4],matStructData[:,5],matStructData[:,6],
                                  matStructData[:,7],matStructData[:,8], theseRunParams)
                             
    return theseData

##
def getPptNum(filename):
    '''Given the filename structure used by the RL code, the ppt number is returned as a string'''
    return str(filename.split('/')[-1].split('_')[0])

##

def makeNewStudyFromFileNamelist(studyName, fileNameList, studyArmDict):
    
    '''Given a name for a study, a list of files and a dictionary that matches the pptnumber to a study arm,
        a study object is created that has multiple participant objects, and each of those have associated sessions.
        From this study requests for fitting and averaging over individual sessions or trial types can be made'''
    
    fileList = np.loadtxt(fileNameList,unpack=True,dtype='string')
    
    theStudy=Study(studyName)
    
    for filename in fileList:
        pptNum = getPptNum(filename)
        aSession=readReturnSession(filename)
        
        
        if pptNum in theStudy.ppts:
            print "adding new session ",aSession.sessionName," on existing ppt ",pptNum," to study ",theStudy.studyName
            theStudy.ppts[pptNum].addSession(aSession)
        else:
            print "adding new session ",aSession.sessionName," and new ppt ",pptNum," to study ",theStudy.studyName
            theStudy.addNewPpt(pptNum, studyArmDict[pptNum],aSession)


    return theStudy


##

def makeNewStudyFromDirectory(studyName, fileNameDir, studyArmDict):
    
    '''Given a name for a study, a directory containing files and a dictionary that matches the pptnumber to a study arm,
        a study object is created that has multiple participant objects, and each of those have associated sessions.
        From this study requests for fitting and averaging over individual sessions or trial types can be made'''
    
    fileList = glob.glob(fileNameDir+'*.mat')
    
    theStudy=Study(studyName)
    
    for filename in fileList:
        pptNum = getPptNum(filename)
        aSession=readReturnSession(filename)
        
        
        if pptNum in theStudy.ppts:
            print "adding new session ",aSession.sessionName," on existing ppt ",pptNum," to study ",theStudy.studyName
            theStudy.ppts[pptNum].addSession(aSession)
        else:
            print "adding new session ",aSession.sessionName," and new ppt ",pptNum," to study ",theStudy.studyName
            theStudy.addNewPpt(pptNum, studyArmDict[pptNum],aSession)


    return theStudy


##

def getTimeFeedbackShown(thisSession, trialType, feedback):
    '''
        given a session, trial type (gain/look/loss/all) and a feedback (seen pound coin with "gain/look/loss" = +1, seen "nothing" -1, or all)
        function returns the times in milliseconds from the first real scan specified by the session's ti feedback is shown.
        '''
    
    if trialType == 'gain':
        trials=thisSession.getGainTrials()
    elif trialType == 'look':
        trials=thisSession.getLookTrials()
    elif trialType == 'loss':
        trials=thisSession.getLossTrials()
    elif trialType == 'all':
        trials=thisSession.trial
    else:
        raise ValueError("choose trialType to be gain, look or loss or all")
    
    if feedback != 'all':
        #pick out feedback
        trials = np.where(thisSession.feedback[trials] == feedback)
    
    
    return thisSession.timeStimPresn[trials] - thisSession.runParams.ti + thisSession.runParams.choiceTime + thisSession.runParams.stimuliTime

##

def getTimeStimulusShown(thisSession, trialType):
    '''
        given a session and trial type (gain/look/loss) function returns the times in milliseconds from the first real scan
        specified by the session's ti the stimulus is shown.
        '''
    
    if trialType == 'gain':
        trials=thisSession.getGainTrials()
    elif trialType == 'look':
        trials=thisSession.getLookTrials()
    elif trialType == 'loss':
        trials=thisSession.getLossTrials()
    elif trialType == 'all':
        trials=thisSession.trial
    else:
        raise ValueError("choose trialType to be gain, look or loss or all")
    
    
    return thisSession.timeStimPresn[trials] - thisSession.runParams.ti



##

def findProbabilityAndPredictionError(thisSession, alpha, beta, rewardWeight, trialType, plot=False):
    
    '''
        given session, this code calculates the probability of a choice and the reward prediction error associated
        with each trial in the session for a given trial type (gain, look, loss).
        This part of the code heavily based on Mathias Pessiglione's matlab/SPM code '''
    
    
    #get wanted trials only
    if trialType == 'gain':
        trials=thisSession.getGainTrials()
    elif trialType == 'look':
        trials=thisSession.getLookTrials()
    elif trialType == 'loss':
        trials=thisSession.getLossTrials()
    else:
        raise ValueError("choose trialType to be gain, look or loss")

    #zero Qa and Qb
    
    qA,qB=(0.,0.)
    #probabilities corresponding to the chosen action
    proba=np.zeros(len(trials))
    error=np.zeros(len(trials))


    for i, t in enumerate(trials):
        pA = np.exp((qA/beta))/(np.exp((qA/beta))+np.exp((qB/beta)))
        pB = np.exp((qB/beta))/(np.exp((qA/beta))+np.exp((qB/beta)))
        
        feedback=thisSession.feedback[t] #did they see the gain look loss (+1) or did they see nothing (-1)
        response=thisSession.response[t] #did they pick the image associated with gain look loss 80% of time
        
        if response == 1:  #ppt choses image picture (gain look loss)
            proba[i] = np.log(pA)
            error[i] = rewardWeight*feedback - qA #if saw what was expected 80% of the time rW*f=1
            qA = qA + alpha*error[i]

        else:              #ppt choses nothing picture
            proba[i] = np.log(pB)
            error[i] = rewardWeight*feedback - qB #if saw what was expected 80% of the time rW*f=1,
            qB = qB + alpha*error[i]

        if plot == True:
            plt.figure()
            plt.title(trialType+" trials for ppt "+ppt+" session "+sessionName)
            plt.plot(np.arange(len(proba)),proba,'ro', linestyle='--',label="probability")
            plt.plot(np.arange(len(proba)),error,'bx', linestyle=':', label="reward p.e.")
            plt.legend(loc='best')
            plt.show()
    
    return proba, error



##


def naiveParameterFitting(thisStudy, alphasRange, betasRange, rewardWeightsRange, numAlphas=50, numBetas=50, numRewW=20, plot=False):
    '''Attempts to fit a range of alpha, beta and reward weights to all the sessions in a study. No interpolation or clever parameter fitting, just finds a parameter space and picks the alpha, beta and reward that returns the highest log-likelihood value and assigns it to the session. If you want one parameter fixed, put that value in variableRange and specify numVariables to 1'''
    
    if numAlphas != 1:
        alphas = np.linspace(alphasRange[0], alphasRange[1], numAlphas)
    else:
        alphas=alphasRange
    if numBetas != 1:
        betas  = np.linspace( betasRange[0],  betasRange[1],  numBetas)
    else:
        betas=betasRange
    if numRewW != 1:
        rewardWeights = np.linspace(rewardWeightsRange[0], rewardWeightsRange[1], numRewW)
    else:
        rewardWeights = rewardWeightsRange


    logLikelihoods = np.zeros((numRewW,numAlphas,numBetas))
    for ppt in thisStudy.listPpts():
        for sessionName in thisStudy.ppts[ppt].listSessions():
            #single out this session only
            thisSession=thisStudy.ppts[ppt].sessions[sessionName]
            
            for trialType in ['gain', 'look','loss']:
                #loop over the parameter spaces
                for r, rewardWeight in enumerate(rewardWeights):
                    for a, alpha in enumerate(alphas):
                        for b, beta in enumerate(betas):
                            proba, error = findProbabilityAndPredictionError(thisSession, alpha, beta, rewardWeight, trialType, plot=False)
                            logLikelihoods[r,a,b] = np.sum(proba)
                
                
                    # show LL image for each session for each reward to see the parameter space
                    if plot==True:
                        plt.figure()
                        plt.title("reward: "+str(rewardWeight)+" ppt: "+ppt+" session: "+sessionName+" for trial type "+trialType)
                        plt.imshow(-np.log(-logLikelihoods[r,:,:]), extent=[np.min(alphas),np.max(alphas),np.min(betas), np.max(betas)],interpolation='nearest')
                        plt.xlabel('alpha')
                        plt.ylabel('beta')
                        plt.colorbar()
                        plt.show()


                bestRewWInd,bestAlphaInd,bestBetaInd = np.where(logLikelihoods == np.max(logLikelihoods))
                thisSession.alphas[trialType] = alphas[bestAlphaInd]
                thisSession.betas[trialType]= betas[bestBetaInd]
                thisSession.rewardWeights[trialType] = rewardWeights[bestRewWInd]


