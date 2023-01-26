import os
import soundfile as sf







source_base_path = r'D:\Project\sound_data\korean_voice\data\remote\PROJECT\AI학습데이터\KoreanSpeech\data\1.Training\2.원천데이터'
noise_base_path = r'D:\Project\sound_data\LibriMix\data\wham_noise\tr'

count = 0

# source_path_list = []
# noise_path_list = []
f = open('./clean.txt', 'w')
for root, dirs, files in os.walk(source_base_path):
    for filename in files:
        if filename.endswith('.wav'):
            file_path = os.path.join(root, filename)
            f.write(f'{file_path}\n')
            # source_path_list.append(file_path)
            count +=1

print(count)

f = open('./noise.txt', 'w')
count = 0
for root, dirs, files in os.walk(noise_base_path):
    for filename in files:
        if filename.endswith('.wav'):
            file_path = os.path.join(root, filename)
            f.write(f'{file_path}\n')
            # noise_path_list.append(file_path)
            count +=1

print(count)