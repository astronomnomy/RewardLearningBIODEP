from rewardlearning import *

###################################################

################## dictionaries ###################
#pptnumber : study arm
# study arms:
# 1: control no treatment
# 2: HAM-D < 7 on medication
# 3: HAM-D > 13 on medication
# 4: HAM-D > 17 no medication
studyArmDict={
                '100':1,
                '5004':1,
                '5005':1,
                '5006':1,
            }

###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################


#.txt file with all the .mat full path names that are being analysed.
# OR use every .mat file in a directory

matfiles="/Users/cc401/data/TH2/Oxford/task/RL/filelist.txt"
matfilesDir="/Users/cc401/data/TH2/Oxford/task/RL/"


#theStudy=makeNewStudyFromFileNamelist('BIODEP', matfiles, studyArmDict)
theStudy=makeNewStudyFromDirectory('BIODEP', matfilesDir, studyArmDict)


####### TO GET THE TIMES OUT FOR THE STUDY ########
#SOME EXAMPLES
trialTypeWanted='gain'
feedbackWanted=1
pptWanted='5004'
sessionWanted='1'
thisSession=theStudy.ppts[pptWanted].sessions[sessionWanted]

timesAllStimulusShown = getTimeStimulusShown(thisSession, 'all')
print timesAllStimulusShown

timesSomeStimulusShown = getTimeStimulusShown(thisSession, trialTypeWanted)
print timesSomeStimulusShown

timesAllFeedbackShown = getTimeFeedbackShown(thisSession, 'all', 'all')
print timesAllFeedbackShown

timesSomeFeedbackShown = getTimeFeedbackShown(thisSession, trialTypeWanted, feedbackWanted)
print timesSomeFeedbackShown

print theStudy.studyName

################## do naive fitting ####################
###### Pretty sure the fitting is actually bugged ######
########## but this is how you call it anyway ##########

alphasRange = [0.01,0.5]
betasRange  = [0.01,0.5]
rewardWeightsRange = [0.05, 2]
numAlphas = 20
numBetas = 20
numRewW = 5


naiveParameterFitting(theStudy, alphasRange, betasRange, rewardWeightsRange, numAlphas=numAlphas, numBetas=numBetas, numRewW=numRewW, plot=False)

theStudy.printResults()

