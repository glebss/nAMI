import soundfile as sf
import os
from tqdm import tqdm
import numpy as np

thrs = 0.0001
raw_noises_folder = '../raw_noises/'
def main():
    pors = []
    with open('./noises_not_used_in_ami.txt') as f:
        not_used_in_ami = [line.rstrip() for line in f.readlines()]
    
    
    for wav_file in tqdm(os.listdir(raw_noises_folder)):
        if wav_file[:-4] not in not_used_in_ami:
            continue
        wav, sr = sf.read(os.path.join(raw_noises_folder, wav_file))
        portion = len(wav[wav**2 <= thrs*np.mean(wav**2)])/len(wav)*100
        pors.append(portion)
        print('for {} silence portion is {:.2f}'.format(wav_file, portion))
    print(np.mean(pors))

if __name__ == '__main__':
    main()