import timeit

'''
* These variables are used to keep track of times needed by each individual component
'''
minTimes,maxTimes,avgTimes = {}, {}, {}
timeKeys = ["OVERALL","DATA_TRANSFER","IMG_CONVERSION","SEGMENT","FACEHIDING","STABILIZE","CLASSIFICATION","INTERFRAME"]
timeStrings = {
    "OVERALL": "Overall:",
    "DATA_TRANSFER": "   Waiting + Data Transfer:",
    "IMG_CONVERSION": "   Image Conversion:",
    "SEGMENT": "   Segmentation:",
    "FACEHIDING": "   Face Hiding:",
    "STABILIZE": "   Stabilizer",
    "CLASSIFICATION": "   Classification:",
    "INTERFRAME": "   Inter-frame difference"
}
for key12 in timeStrings.keys():
    minTimes[key12] = 100
    avgTimes[key12] = 0.0
    maxTimes[key12] = 0



def recordTimings(start_time,time_key,noOfFramesCollected):
    '''
    This performs the manipulation of average, min, max timings for each of the components

    Parameters
    ----------
    start_time : This is the base reference start_time:Timer with reference to which the current time is measured
                and the difference is the time elapsed which is used for calculation of average, min, max timings
    time_key : This indicates the timings of which component need to be updated.
    
    Returns
    -------
    Timer
        This function returns the current instance of timer so that, this can be used in the next invokation of this function.

    '''
    global minTimes,maxTimes,avgTimes
    if noOfFramesCollected != 0: 
        elapsed = timeit.default_timer() - start_time
        avgTimes[time_key] = avgTimes[time_key] * ((noOfFramesCollected-1)/noOfFramesCollected) + elapsed/noOfFramesCollected
        minTimes[time_key] = elapsed if elapsed < minTimes[time_key] else minTimes[time_key]
        maxTimes[time_key] = elapsed if elapsed > maxTimes[time_key] else maxTimes[time_key]
    return timeit.default_timer()


def displayAllTimings(noOfFramesCollected):
    print('\n\nTimings for %d frames'%noOfFramesCollected)
    for key12 in timeKeys: 
        print(timeStrings[key12])
        print('          Min Time taken:',"%.4fs"%minTimes[key12])
        print('          Avg Time taken:',"%.4fs"%avgTimes[key12])
        print('          Max Time taken:',"%.4fs"%maxTimes[key12])

