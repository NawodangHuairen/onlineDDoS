import yaml
import sys
import numpy as np
import hashlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from scipy import stats
from datetime import datetime, timedelta


def getAgentHash(agent, agentHashRange) :
    hashret = int(hashlib.sha1(agent.encode('utf-8')).hexdigest(), 16) % agentHashRange

    return str(hashret)

def loadConfig():
    with open(sys.argv[1], "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg

def getHash(fullURI, queryHashRange):
    hashret = 0

    #Handles cases with no '/'
    if(fullURI.find('/') == -1) :
        fullURI = '/' + fullURI

    uri = fullURI.split('?', 1)[0]

    if (len(uri.rsplit('/', 1)) > 1) :
        request = uri.rsplit('/', 1)[1]
    else :
        request = ''

    if (len(fullURI.split('?', 1)) > 1) :
        request = request + '?' + fullURI.split('?', 1)[1]
    else :
        request = request

    hashret = int(hashlib.sha1(request.encode('utf-8')).hexdigest(), 16) % queryHashRange
    uri = uri.rsplit('/', 1)[0]

    return str(hashret)

def getURI(fullURI):
    uri = fullURI.split('?', 1)[0]
    uri = uri.rsplit('/', 1)[0]


    if uri == '' :
        return '<EMPTY>'
    else :
        return str(uri)

def converttodatetime(x, seqlen):
    # if len(x)< 20:
    if len(x)< seqlen:    # add
        x += '.000'
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")

#TimeDif 0 Padding
# Start = 14
# End = 15
def getTimeClass(time_dif):
    time = 0
    if time_dif == 0.0:
        time = 0
    elif time_dif < 0.001 :
        time = 1
    elif time_dif < 0.05 :
        time = 2
    elif time_dif < 0.1 :
        time = 3
    elif time_dif < 1 :
        time = 4
    elif time_dif < 15 :
        time = 5
    elif time_dif < 180 :
        time = 6
    elif time_dif > 300 :
        time = 7

    return str(time)

#Source IP  Dest IP Time    Packet Len  IP Flags    TCP Len TCP Ack TCP Flags   TCP Window Size UDP Len ICMP Type   Protocols   Highest Layer(Protocol) Info
def prepDataFrame(df, agentHashRange, queryHashRange) :
    #create a deepcopy of the original df
    df_temp = df.copy()
    if type(df_temp['Absolute Time'].iloc[0]) == str:
        df_temp['Absolute Time'] = df_temp['Absolute Time'].apply(lambda x: converttodatetime(x, config['SEQUENCELENGTH'])) # add

    #converting type of status
    df_temp = df_temp.astype({'Source IP' : 'str'})
    df_temp = df_temp.astype({'Dest IP' : 'str'})
    # df_temp = df_temp.astype({'Relative Time' : 'str'})
    df_temp = df_temp.astype({'Packet Len': 'str'})
    df_temp = df_temp.astype({'IP Flags' : 'str'})
    df_temp = df_temp.astype({'TCP Len' : 'str'})
    df_temp = df_temp.astype({'TCP Flags' : 'str'})
    df_temp = df_temp.astype({'TCP Window Size': 'str'})
    df_temp = df_temp.astype({'Protocols': 'str'})
    df_temp = df_temp.astype({'Highest Layer(Protocol)': 'str'})
    df_temp = df_temp.astype({'Info' : 'str'})

    ##TCP ACK Feature Engineering
    tcpack = np.array(df_temp['TCP Ack'])
    bin_edges = stats.mstats.mquantiles(tcpack, np.arange(0,1,0.01)[1:])

    df_temp.loc[df_temp['TCP Ack'] <= bin_edges[int(0)], 'new TCP Ack'] = int(0)

    for i in range(len(bin_edges)-1):
        df_temp.loc[(bin_edges[int(i)] < df_temp['TCP Ack']) & (df_temp['TCP Ack'] <= bin_edges[int(i+1)]), 'new TCP Ack'] = int(i+1)

    df_temp.loc[df_temp['TCP Ack'] > bin_edges[-1], 'new TCP Ack'] = int(len(bin_edges))
    df_temp['new TCP Ack'] = df_temp['new TCP Ack'].astype('str')

    #Relative Time Feature Engineering
    # reltime = np.array(df['Relative Time'])
    # round_reltime = np.around(reltime, decimals=0, out=None)
    # df_temp.insert(1, "new Relative Time", round_reltime, True)
    # df_temp['new Relative Time'] = df_temp['new Relative Time'].astype('str')

    #calculate the difference between requests from a specific user
    df_temp['time_diff'] = df_temp.groupby('Source IP')['Absolute Time'].diff()

    #Maybe can remove this (DOUBLE CHECK)
    df_temp['time_diff_group'] = df_temp['time_diff'].apply(lambda x: getTimeClass(x.total_seconds()))

    #Calculate the Combined Input of URI Hash Status Time
    df_temp['Input'] = df_temp['Packet Len'] + '<JOIN>' + df_temp['IP Flags'] + '<JOIN>' + df_temp['TCP Len'] + '<JOIN>' + df_temp['new TCP Ack'] + '<JOIN>' + df_temp['TCP Flags'] + '<JOIN>' + df_temp['TCP Window Size'] + '<JOIN>' + df_temp['Highest Layer(Protocol)']

    return df_temp

def getSignificantRequest(dataframe, hashThreshold) :
    freq = dataframe['Input'].value_counts(normalize=True)

    ret = []

    index = freq.index
    for i in range(len(freq)):
        if freq[i] > hashThreshold :
            ret.append(index[i])

    return ret

def keepOrHash(uri, sig, inputHashRange) :
    if uri in sig :
        return uri
    else :
        return str(int(hashlib.sha1(uri.encode('utf-8')).hexdigest(), 16) % inputHashRange)

def sequentializeDataFrame(df, sig, inputHashRange, seqlen): # add
    #create a deepcopy of the original df
    df_temp = df.copy()
    if type(df_temp['Absolute Time'].iloc[0]) == str:
        df_temp['Absolute Time'] = df_temp['Absolute Time'].apply(lambda x: converttodatetime(x, config['SEQUENCELENGTH'])) # add
    #converting type of status
    df_temp = df_temp.astype({'Source IP' : 'str'})
    df_temp = df_temp.astype({'Dest IP' : 'str'})
    # df_temp = df_temp.astype({'Relative Time' : 'str'})
    df_temp = df_temp.astype({'Packet Len': 'str'})
    df_temp = df_temp.astype({'IP Flags' : 'str'})
    df_temp = df_temp.astype({'TCP Len' : 'str'})
    df_temp = df_temp.astype({'TCP Ack' : 'str'})
    df_temp = df_temp.astype({'TCP Flags' : 'str'})
    df_temp = df_temp.astype({'TCP Window Size': 'str'})
    df_temp = df_temp.astype({'Protocols': 'str'})
    df_temp = df_temp.astype({'Highest Layer(Protocol)': 'str'})
    df_temp = df_temp.astype({'Info' : 'str'})
    df_temp = df_temp.astype({'Input': 'str'})

    #Uncomment if you want to hash the Input
    df_temp['Input'] = df_temp['Input'].apply(lambda x: keepOrHash(x, sig, inputHashRange))

    #create groups based on 1 min interval
    df_temp['groups'] = df_temp.groupby('Source IP')['time_diff'].apply(lambda x: x.gt(pd.Timedelta(1, 'm')).cumsum())
    df_temp['time_diff'] = df_temp['time_diff'].apply(lambda x: getTimeClass(x.total_seconds()))
    #grouping in sequences of 20 length
    df_temp['group_len'] = df_temp.groupby(['Source IP', 'Dest IP', 'groups'])['Absolute Time'].rank(method = 'first')
    # df_temp['group_len'] = df_temp['group_len'].apply(lambda x: math.ceil(x/20))
    df_temp['group_len'] = df_temp['group_len'].apply(lambda x: math.ceil(x/seqlen)) # add

    #create groups based on "remote_addr" and "groups"
    df_temp = df_temp.groupby(['Source IP', 'Dest IP', 'groups', 'group_len'])

    #aggregation
    sr = df_temp['Protocols', 'Info'].agg(lambda x: "<SEP>".join(x))
    sr['Input'] = df_temp['Input'].agg(lambda x: "<SEP>".join(x))
    sr.reset_index(inplace=True)
    #converting to dataframe
    #sr = sr.to_frame()
    sr = sr.drop(columns = ['groups', 'group_len'])
    return sr

def getCountryAgentPair(x, y):
    agent = x.split('<SEP>')[0]
    country = y.split('<SEP>')[0]

    return str(country) + '<JOIN>' + str(agent)

def getFirstOnly(y):
    country = y.split('<SEP>')[0]

    return str(country)


#Remove request if Remote Addresses (IP) appear 10 or less times
#Because the data is not in chronological form, sort by timestamp to get in chronological form
def filterAndSort(df) :
    df = df.sort_values(by=['Absolute Time'])
    before = len(df)
    df = df.groupby('Source IP').filter(lambda x: len(x) > 10)
    after = len(df)
    df = df.reset_index()

    print("Before = {}, After = {}".format(before, after))

    return df

#Split dataframe into 5s interval
def getIntervals(df) :
    ret = []
    S = pd.to_datetime(df['Absolute Time'])
    timebins = (S - S[0]).astype('timedelta64[m]')
    timebins = timebins.tolist()
    timebins = [math.floor(time/1) for time in timebins]
    timebins = [pd.Series(timebins)]

    for i, g in df.groupby(timebins):
        ret.append(g.reset_index(drop=True))

    return ret

def main():
    print("*****     Starting Preprocessing     ******")
    config = loadConfig()

    dfA = pd.read_csv(config['datapath']['a'], parse_dates=['Absolute Time'])
    dfN1 = pd.read_csv(config['datapath']['n1'], parse_dates=['Absolute Time'])

    # #load N2
    # dfN2 = pd.read_csv(config['datapath']['n2'], parse_dates=['Absolute Time'])

    #Remove request if Remote Addresses (IP) appear 10 or less times
    #Because the data is not in chronological form, sort by timestamp to get in chronological form
    dfA = filterAndSort(dfA)
    dfN1 = filterAndSort(dfN1)
    # dfN2 = filterAndSort(dfN2)

    #Prepares A1 and N1 for sequentializing
    df_normal = prepDataFrame(dfN1, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
    # df_normal2 = prepDataFrame(dfN2, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
    significantNormal = getSignificantRequest(df_normal, config['variablesHash']['inputHashThreshold'])

    #Sequentializes A1 and N1
    df_normal = sequentializeDataFrame(df_normal, significantNormal, config['variablesHash']['inputHashRange'], config['SEQUENCELENGTH']) # add
    # df_normal2 = sequentializeDataFrame(df_normal2, significantNormal, config['variablesHash']['inputHashRange'], config['SEQUENCELENGTH']) # add

    #Prepares the Histogram for Agent and Country
    df_normal['Histo'] = df_normal.apply(lambda row: getCountryAgentPair(row['Protocols'], row['Info']), axis=1)
    # df_normal2['Histo'] = df_normal2.apply(lambda row: getCountryAgentPair(row['Protocols'], row['Info']), axis=1)

    #Get Agent
    df_normal['Protocols'] = df_normal.apply(lambda row: getFirstOnly(row['Protocols']), axis=1)
    # df_normal2['Protocols'] = df_normal2.apply(lambda row: getFirstOnly(row['Protocols']), axis=1)

    #Get Country
    df_normal['Info'] = df_normal.apply(lambda row: getFirstOnly(row['Info']), axis=1)
    # df_normal2['Info'] = df_normal2.apply(lambda row: getFirstOnly(row['Info']), axis=1)


    #Save the dataframe as artefacts.
    df_normal.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'N1.csv', index = None, header=True)

    df_attack_intervals = getIntervals(dfA)

    # if config['ONLINETRAINING'] :
    count = 0
    for df_int in df_attack_intervals[:-config['metadata']['attackintervals']] :
        df_attack = prepDataFrame(df_int, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
        df_attack = sequentializeDataFrame(df_attack, significantNormal, config['variablesHash']['inputHashRange'], config['SEQUENCELENGTH']) # add
        df_attack['Histo'] = df_attack.apply(lambda row: getCountryAgentPair(row['Protocols'], row['Info']), axis=1)
        df_attack['Protocols'] = df_attack.apply(lambda row: getFirstOnly(row['Protocols']), axis=1)
        df_attack['Info'] = df_attack.apply(lambda row: getFirstOnly(row['Info']), axis=1)
        df_attack.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_' + str(count) + '.csv', index = None, header=True)
        count = count + 1
    # else :
    df_attack_batch = []
    # count = 0
    for df_int in df_attack_intervals[:-config['metadata']['attackintervals']] :
        df_attack = prepDataFrame(df_int, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
        df_attack = sequentializeDataFrame(df_attack, significantNormal, config['variablesHash']['inputHashRange'], config['SEQUENCELENGTH']) # add
        df_attack['Histo'] = df_attack.apply(lambda row: getCountryAgentPair(row['Protocols'], row['Info']), axis=1)
        df_attack['Protocols'] = df_attack.apply(lambda row: getFirstOnly(row['Protocols']), axis=1)
        df_attack['Info'] = df_attack.apply(lambda row: getFirstOnly(row['Info']), axis=1)
        df_attack_batch.append(df_attack)
        # count = count + 1

    df_attack = pd.concat(df_attack_batch)
    df_attack.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A1_full.csv', index = None, header=True)


    # df_normal2.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'N2.csv', index = None, header=True) # add
    df_test = [] 
    # #N2 test set
    # df_test.append(df_normal2) # remove?

    # A2 test set
    for df_int in df_attack_intervals[-config['metadata']['attackintervals']:] :
        df_attack = prepDataFrame(df_int, config['variablesHash']['agentHashRange'], config['variablesHash']['queryHashRange'])
        # df_attack.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'A2_' + str(count) + '(PreSequentialize).csv', index = None, header=True)
        df_attack = sequentializeDataFrame(df_attack, significantNormal, config['variablesHash']['inputHashRange'], config['SEQUENCELENGTH']) # add
        df_attack['Histo'] = df_attack.apply(lambda row: getCountryAgentPair(row['Protocols'], row['Info']), axis=1)
        df_attack['Protocols'] = df_attack.apply(lambda row: getFirstOnly(row['Protocols']), axis=1)
        df_attack['Info'] = df_attack.apply(lambda row: getFirstOnly(row['Info']), axis=1)
        df_test.append(df_attack)

    df_test = pd.concat(df_test) 
    df_test.to_csv(r'' + config['metadata']['uniqueID'] + '/' + config['metadata']['artefact'] + '/' + 'TEST.csv', index = None, header=True)
    
    
    print("*****     Ending Preprocessing     ******")

if __name__ == "__main__":
    main()
