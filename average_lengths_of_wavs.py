import soundfile as sf
import os
from tqdm import tqdm

raw_noises_folder = '../raw_noises/'
def main():
    with open('./noises_not_used_in_ami.txt') as f:
        not_used_in_ami = [line.rstrip() for line in f.readlines()]
    
    d = {}
    for wav_file in tqdm(os.listdir(raw_noises_folder)):
        if wav_file[:-4] not in not_used_in_ami:
            continue
        wav, sr = sf.read(os.path.join(raw_noises_folder, wav_file))
        if wav_file.split('_')[0] in d.keys():
            d[wav_file.split('_')[0]][0] += 1
            d[wav_file.split('_')[0]][1] += len(wav)
        else:
            d[wav_file.split('_')[0]] = [0,0]
            d[wav_file.split('_')[0]][0] = 1
            d[wav_file.split('_')[0]][1] = len(wav)
    
    for key in d.keys():
        print(key, '{:.2f}'.format(d[key][1]/d[key][0]/sr))

if __name__ == '__main__':
    main()
            
        
        
    