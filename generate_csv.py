import os
import math
import mne
import numpy as np

rootdir = "C:\\Users\\marcu\\Documents\\CHB-MIT\\"
segment_length = 10
folder = []
summary = []
summarycontent = ""
seizurecount_dic = {}
seizuretiming_dic = {}
seizuretuple_dic = {}

for dir in os.listdir("."):
    if dir[:3] == "chb":
        folder.append(dir)

for x in folder:
    for y in os.listdir(f".\\{x}"):
        if y[-3:] == "txt":
            summary.append(f"{x}\\{y}")

for x in range(len(summary)):
    with open(f".//{summary[x]}", "r") as file:
        lines = file.readlines()
        timingstring = []
        filename = ""
        for y in range(len(lines)):
            line = lines[y]
            if line[:len("File name")] == "File Name":
                if len(timingstring) != 0:
                    seizuretiming_dic[filename] = timingstring
                string = line.strip()
                index = string.find("chb")
                filename = string[index:]
                timingstring = []
            elif line[:len("Number")] == "Number":
                string = line.strip()
                index = string.find(":")
                number = string[index+2:]
                seizurecount_dic[filename] = number
            elif line[:len("Seizure")] == "Seizure":
                string = line.strip()
                start_index = string.find(":") + 2
                end_index = string.find("second") - 1
                timingstring.append(string[start_index:end_index])
            if y == len(lines)-1 and len(timingstring) != 0:
                seizuretiming_dic[filename] = timingstring

for index, (key, value) in enumerate(seizuretiming_dic.items()):
    segment = []
    for x in range(math.floor(len(value)/2)):
        segment.append((int(value[2*x]), int(value[2*x+1])))
    seizuretuple_dic[key] = segment

for index, (key, value) in enumerate(seizuretuple_dic.items()):
    print(key)
    print(f"{index + 1}. {value}")
# for index, (key, value) in enumerate(seizurecount_dic.items()):
#     print(f'{index}. File {key} Seizure count: {value}')

# print(len(seizurecount_dic))


def overlap(start1, end1, start2, end2):
    return end1 > start2 and end2 > start1


for x in range(len(folder)):
    print(f"{x}. {folder[x]}")


for p in range(len(folder)):
    seizurenparrays = []
    normalnparrays = []
    print(f"Processing: Patient{p}")
    seizurefiles = []
    nonseizurefiles = []
    edf = os.listdir(f".//{folder[p]}")
    for x in edf:
        if x[-len("seizures"):] == "seizures":
            seizurefiles.append(x[:len(x)-len("seizures")-1])
    for x in edf:
        if x not in seizurefiles and x[-len("edf"):] == "edf":
            nonseizurefiles.append(x)

    for x in nonseizurefiles:
        edf = mne.io.read_raw_edf(f".//{folder[p]}//{x}", preload=False)
        samplecount = int(math.ceil(edf.times[-1]/segment_length))
        for y in range(samplecount):
            if (y+1)*segment_length <= edf.times[-1]:
                subset = edf.copy().crop(y*segment_length, (y+1)*segment_length, False).get_data()
                normalnparrays.append(subset)

    for x in range(len(seizurefiles)):
        edf = mne.io.read_raw_edf(f".//{folder[p]}//{seizurefiles[x]}", preload=False)
        Timings = seizuretuple_dic[seizurefiles[x]]
        samplecount = int(math.ceil(edf.times[-1]/segment_length))
        for y in range(samplecount):
            if (y+1)*segment_length <= edf.times[-1]:
                Ictal = False
                subset = edf.copy().crop(y*segment_length, (y+1)*segment_length, False).get_data()
                for z in Timings:
                    if overlap(z[0], z[1], y*segment_length, (y+1)*segment_length):
                        Ictal = True
                if Ictal:
                    seizurenparrays.append(subset)
                else:
                    normalnparrays.append(subset)

seizuretuples = tuple(seizurenparrays)
normaltuples = tuple(normalnparrays)

seizurearrays = np.vstack(seizuretuples)
normalarrays = np.vstack(normaltuples)

print(seizurearrays.shape)
print(normalarrays.shape)

np.savetxt("Ictal.csv", seizurearrays, delimiter=",")
np.savetxt("Non-Ictal.csv", normalarrays, delimiter=",")
