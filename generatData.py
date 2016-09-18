import numpy as np
import preprocessing as pre

#paths to data and labels
pathData0 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/LTU/*16k.wav'
pathData1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/CNE/*16k.wav'
pathData2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/GEN/*16k.wav'
pathData3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/HRO/*16k.wav'
pathData4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/ICO/*16k.wav'
pathData5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/KTA/*16k.wav'
pathData6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/LHE/*16k.wav'
pathData7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MDU/*16k.wav'
pathData8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/SCO/*16k.wav'

pathData9 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MKE/*16k.wav'
pathData10 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MMI/*16k.wav'
pathData11 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MPE/*16k.wav'
pathData12 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/MZI/*16k.wav'
pathData13 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/NNE/*16k.wav'
pathData14 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/VZI/*16k.wav'
pathData15 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/audio_data/LTH/*16k.wav'


pathLabel0 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_LTU.csv'
pathLabel1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_CNE.csv'
pathLabel2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_GEN.csv'
pathLabel3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_HRO.csv'
pathLabel4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_ICO.csv'
pathLabel5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_KTA.csv'
pathLabel6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_LHE.csv'
pathLabel7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MDU.csv'
pathLabel8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_SCO.csv'

pathLabel9 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MKE.csv'
pathLabel10 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MMI.csv'
pathLabel11 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MPE.csv'
pathLabel12 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_MZI.csv'
pathLabel13 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_NNE.csv'
pathLabel14 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_VZI.csv'
pathLabel15 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Labels_as_csv/labels_LTH.csv'



pathTimes0 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_LTU.csv'
pathTimes1 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_CNE.csv'
pathTimes2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_GEN.csv'
pathTimes3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_HRO.csv'
pathTimes4 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_ICO.csv'
pathTimes5 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_KTA.csv'
pathTimes6 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_LHE.csv'
pathTimes7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MDU.csv'
pathTimes8 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_SCO.csv'

pathTimes9 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MKE.csv'
pathTimes10 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MMI.csv'
pathTimes11 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MPE.csv'
pathTimes12 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_MZI.csv'
pathTimes13 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_NNE.csv'
pathTimes14 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_VZI.csv'
pathTimes15 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/StartEndTime_as_csv/startEnd_time_LTH.csv'



pathBetterLabel2 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_GEN.csv'
pathBetterLabel3 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_HRO.csv'
pathBetterLabel11 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_MPE.csv'
pathBetterLabel15 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_LTH.csv'
pathBetterLabel0 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_LTU.csv'
pathBetterLabel7 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_MDU.csv'
pathBetterLabel9 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_MKE.csv'
pathBetterLabel12 = '/home/huynh-tan/Dokumente/Bachelor_Thesis/Labels/Bette_Labels_as_csv/labels_speaker_MZI.csv'




#parameters for preprocessing
sampleRate = 16000
maxSentenceStep = 9*sampleRate #9sec longest sentence
maxWordStep = int(0.3*sampleRate) #0.3sec mean word time
newLabels = 1



#reading the data and preprocessing
if newLabels == 0:
    examples0, labels0 = pre.makeInputSeq(pathData0, pathTimes0, pathLabel0, maxSentenceStep, maxWordStep, sampleRate)
    examples1, labels1 = pre.makeInputSeq(pathData1, pathTimes1, pathLabel1, maxSentenceStep, maxWordStep, sampleRate)
    examples2, labels2 = pre.makeInputSeq(pathData2, pathTimes2, pathLabel2, maxSentenceStep, maxWordStep, sampleRate)
    examples3, labels3 = pre.makeInputSeq(pathData3, pathTimes3, pathLabel3, maxSentenceStep, maxWordStep, sampleRate)
    examples4, labels4 = pre.makeInputSeq(pathData4, pathTimes4, pathLabel4, maxSentenceStep, maxWordStep, sampleRate)
    examples5, labels5 = pre.makeInputSeq(pathData5, pathTimes5, pathLabel5, maxSentenceStep, maxWordStep, sampleRate)
    examples6, labels6 = pre.makeInputSeq(pathData6, pathTimes6, pathLabel6, maxSentenceStep, maxWordStep, sampleRate)
    examples7, labels7 = pre.makeInputSeq(pathData7, pathTimes7, pathLabel7, maxSentenceStep, maxWordStep, sampleRate)
    examples8, labels8 = pre.makeInputSeq(pathData8, pathTimes8, pathLabel8, maxSentenceStep, maxWordStep, sampleRate)

    examples9, labels9 = pre.makeInputSeq(pathData9, pathTimes9, pathLabel9, maxSentenceStep, maxWordStep, sampleRate)
    examples10, labels10 = pre.makeInputSeq(pathData10, pathTimes10, pathLabel10, maxSentenceStep, maxWordStep, sampleRate)
    examples11, labels11 = pre.makeInputSeq(pathData11, pathTimes11, pathLabel11, maxSentenceStep, maxWordStep, sampleRate)
    examples12, labels12 = pre.makeInputSeq(pathData12, pathTimes12, pathLabel12, maxSentenceStep, maxWordStep, sampleRate)
    examples13, labels13 = pre.makeInputSeq(pathData13, pathTimes13, pathLabel13, maxSentenceStep, maxWordStep, sampleRate)
    examples15, labels15 = pre.makeInputSeq(pathData15, pathTimes15, pathLabel15, maxSentenceStep, maxWordStep, sampleRate)

    test_data, test_labels = pre.makeInputSeq(pathData14, pathTimes14, pathLabel14, maxSentenceStep, maxWordStep, sampleRate)



    examples = np.concatenate((examples0, examples1, examples2, examples3, examples4, examples5, examples6, examples7, examples8
                               , examples9, examples10, examples11, examples12, examples13, examples15))

    labels = np.concatenate((labels0, labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10
                             , labels11, labels12, labels13, labels15))

else:
    examples2, labels2 = pre.makeInputSeq(pathData2, pathTimes2, pathLabel2, maxSentenceStep, maxWordStep, sampleRate,
                                          pathBetterLabel2, newLabels)
    examples3, labels3 = pre.makeInputSeq(pathData3, pathTimes3, pathLabel3, maxSentenceStep, maxWordStep, sampleRate,
                                          pathBetterLabel3, newLabels)
    examples11, labels11 = pre.makeInputSeq(pathData11, pathTimes11, pathLabel11, maxSentenceStep, maxWordStep,
                                            sampleRate, pathBetterLabel11, newLabels)
    examples0, labels0 = pre.makeInputSeq(pathData0, pathTimes0, pathLabel0, maxSentenceStep, maxWordStep, sampleRate,
                                          pathBetterLabel0, newLabels)
    examples7, labels7 = pre.makeInputSeq(pathData7, pathTimes7, pathLabel7, maxSentenceStep, maxWordStep, sampleRate,
                                          pathBetterLabel7, newLabels)
    examples9, labels9 = pre.makeInputSeq(pathData9, pathTimes9, pathLabel9, maxSentenceStep, maxWordStep, sampleRate,
                                          pathBetterLabel9, newLabels)
    examples15, labels15 = pre.makeInputSeq(pathData15, pathTimes15, pathLabel15, maxSentenceStep, maxWordStep, sampleRate,
                                   pathBetterLabel15, newLabels)


    test_data, test_labels = pre.makeInputSeq(pathData12, pathTimes12, pathLabel12, maxSentenceStep, maxWordStep,
                                            sampleRate, pathBetterLabel12, newLabels)

    useTest = 0
    if useTest is 1:
        'here using a speaker which is not in training data to test'
        examples = np.concatenate(
            (examples0, examples2, examples3, examples7, examples9, examples11, examples15))

        labels = np.concatenate(
            (labels0, labels2, labels3, labels7, labels9, labels11, labels15))
    else:
        'here exclude a part of the data to test'
        examples = np.concatenate(
            (examples0, examples2, examples3, examples7, examples9, examples11, examples15, test_data))

        labels = np.concatenate(
            (labels0, labels2, labels3, labels7, labels9, labels11, labels15, test_labels))


