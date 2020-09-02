import pickle
# import matplotlib.pyplot as plot
import yaml
import pandas as pd
import pickle
import sys
import numpy as np

# Plot and save fig without displaying it on X server 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot



def loadConfig():
    with open(sys.argv[1], "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg

# def filterAndSort(df) :
#     df = df.sort_values(by=['timestamp'])
#     before = len(df)
#     df = df.groupby('remote_addr').filter(lambda x: len(x) > 10)
#     after = len(df)
#     df = df.reset_index()

#     print("Before = {}, After = {}".format(before, after))

#     return df

#Create a function that plots the graphs with acceptance rate of 0.1,0.2 ..... 1.0
#Calculate those who got normal how many were rejected wrongly and not inside the AGT List.
import math

def calculateFalsePositives(agtIPList, agt_attacks, scoreDict, percentages, numNorm) :
    numNormal = numNorm
    numTotalIP = len(scoreDict)
    cutOff = []
    falsepositives = []
    
    for percent in percentages :
        cutOff.append(math.ceil(numTotalIP * percent))
    
    scoreCount = 0
    index = 0
    tp_count = 0 # add
    truepos = [] # add
    
    for (IP, IPD, score) in list(scoreDict.itertuples(index=False, name=None)):
        if IP + IPD in agtIPList:
            scoreCount = scoreCount + 1
        if IP + IPD in agt_attacks: # add
            tp_count = tp_count + 1 # add
            
        index = index + 1
        if index in cutOff :
            falsepositives.append(scoreCount/numNormal)
            truepos.append(tp_count / index) #len(agt_attacks)) # add
            
    return falsepositives, truepos
        
def plotAndSaveGraph(PQ, P, PQTil, Qonline, Qoffline, config, plt=False):
# def plotAndSaveGraph(PQ, P, config):
    percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plot.rcParams['figure.figsize'] = [9, 9]
    fig = plot.figure()
    ax = plot.subplot(111)
    # #ax.plot(percentages, old_graphScoreList, label = "Old P Over Q Scores")
    # #ax.plot(percentages, old_graphPList, label = "Old P Scores")
    ax.plot(percentages, percentages, linewidth=2, label = "Randomized Rejection")
    ax.plot(percentages, P, linewidth=2, label = "N Only")
    ax.plot(percentages, PQ, linewidth=2, label = "Online N/D") #With Transfer")
    ax.plot(percentages, PQTil, linewidth=2, label = "Offline N/D") #Without Transfer") 
#     ax.plot(percentages, Qonline, label = "Online Q")
#     ax.plot(percentages, Qoffline, label = "Offline Q") 
    
#     plot.xlabel('Rejection Threshold', fontsize=24)
#     plot.ylabel('False Reject Rates', fontsize=24)
#     plot.title("False Positive rates for " + config['metadata']['name'])
    ax.legend(fontsize=18)
    if plt:
        plot.savefig(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + '_FPGraph')

def calc_eval(fpr, tpr, agt_norm, agt_attk):
    fpr = np.array(fpr[1:])
    tpr = np.array(tpr)

    Total = len(agt_norm)+len(agt_attk) #len(userScoreP)
    TotalN = len(agt_norm)
    TotalP = Total-TotalN
    FP = fpr*TotalN
    TN = TotalN - FP
    TP = tpr*TotalP
    FN = TotalP-TP
    
    Accr = (TP+TN) / (TP+TN+FP+FN)
    FPR = FP / (FP + TN)
    Prec = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1 = 2 * ((Prec*Rec) / (Prec+Rec))

    return (Accr, FPR, Prec, Rec, F1)



def main():
    print("*****     Starting Evaluation     ******")
    config = loadConfig()

    # #Load User Scores
    userScoreP = pickle.load(open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'PScore', 'rb'))
    userScoreP = userScoreP.sort_values(by = ['P'],ascending=False) # add

    userScorePQ_online = pickle.load(open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'PoverQonline_score', 'rb'))
    userScorePQ_online = userScorePQ_online.sort_values(by = ['PoverQ_online'],ascending=False) # add 

    userScorePQ_offline = pickle.load(open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'PoverQoffline_score', 'rb'))
    userScorePQ_offline = userScorePQ_offline.sort_values(by = ['PoverQ_offline'],ascending=False) 

    userScoreQonline = pickle.load(open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'onlineQ', 'rb'))
    userScoreQonline = userScoreQonline.sort_values(by = ['QWithT'],ascending=False)

    userScoreQoffline = pickle.load(open(config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'offlineQ', 'rb'))
    userScoreQoffline = userScoreQoffline.sort_values(by = ['QWithoutT'],ascending=False)


    agt_normals = []
    agt_attacks = []

    print("Length of userScoreP: ", len(userScoreP))

    for (IP, IPD, score) in list(userScoreP.itertuples(index=False, name=None)):
    #     if IP != '172.16.0.1' and IP == '192.168.10.50': # CIC Friday
    #     if (IP != '172.16.0.1' and IPD=='192.168.10.50') or IP == '192.168.10.50': # CIC Wed
    #     if (IP != '172.16.0.1' and IPD=='192.168.10.50') or IP == '192.168.10.50': 
        if (IP != '172.16.0.1' and  IPD=='192.168.10.50') or IP == '192.168.10.50': 
            agt_normals.append(IP + IPD)
        else:
            agt_attacks.append(IP+IPD)

    print("Length of Normal traffic: ", len(agt_normals))
    print("Length of true attacks: ", len(agt_attacks))

    #Plot some graphs
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    num = len(agt_normals)
    print("Percent: ", num/len(userScoreP))

    graphP, P_tpr = calculateFalsePositives(agt_normals, agt_attacks, userScoreP, percentages, num)
    graphPQ_online, onlinePQ_tpr = calculateFalsePositives(agt_normals, agt_attacks, userScorePQ_online, percentages, num)
    graphPQ_offline, offlinePQ_tpr  = calculateFalsePositives(agt_normals, agt_attacks, userScorePQ_offline, percentages, num)
    graphQonline, _ = calculateFalsePositives(agt_normals, agt_attacks, userScoreQonline, percentages, num)
    graphQoffline, _ = calculateFalsePositives(agt_normals, agt_attacks, userScoreQoffline, percentages, num)   

    graphP.insert(0, 0)
    graphPQ_online.insert(0, 0)
    graphPQ_offline.insert(0, 0)
    # graphQonline.insert(0, 0)
    # graphQoffline.insert(0, 0)

    plotAndSaveGraph(graphPQ_online, graphP, graphPQ_offline, graphQonline, graphQoffline, config, plt=True)

    ### Calculate evaluation metrics 

    P_metrics = np.array(np.transpose(calc_eval(graphP, P_tpr, agt_normals, agt_attacks)))
    onlinePQ_metrics = np.array(np.transpose(calc_eval(graphPQ_online, onlinePQ_tpr, agt_normals, agt_attacks)))
    offlinePQ_metrics = np.array(np.transpose(calc_eval(graphPQ_offline, offlinePQ_tpr, agt_normals, agt_attacks)))

    P_metrics = pd.DataFrame(P_metrics, columns=['Accr_P', 'FPR_P', 'Prec_P', 'Rec_P', 'F1_P'])
    onlinePQ_metrics = pd.DataFrame(onlinePQ_metrics, columns=['Accr_onPQ', 'FPR_onPQ', 
                                                               'Prec_onPQ', 'Rec_onPQ', 'F1_onPQ'])
    offlinePQ_metrics = pd.DataFrame(offlinePQ_metrics, columns=['Accr_offPQ', 'FPR_offPQ', 
                                                                 'Prec_offPQ', 'Rec_offPQ', 'F1_offPQ'])
    df_FPresults = pd.concat([P_metrics,onlinePQ_metrics,offlinePQ_metrics],axis=1)

    df_FPresults.to_csv(config['metadata']['uniqueID'] + '/' + config['metadata']['result'] + 
                        "_evalresults.csv", index=False)


    print("*****     Ending Evaluation     ******")

if __name__ == "__main__":
    main()