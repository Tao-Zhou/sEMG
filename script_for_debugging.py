import os


data_set_path = 'E://srt//CapgMyoData//dbc_proc_python'
dirs = os.listdir(data_set_path)
dir = dirs[1]

files = list(os.walk(data_set_path+'//'+dir))[0][2]
subject_dict = {}
for file in files:
    subject = file[:3]
    if subject in subject_dict.keys():
        subject_dict[subject].append(file)
    else:
        subject_dict[subject] = [file]
print(subject_dict)