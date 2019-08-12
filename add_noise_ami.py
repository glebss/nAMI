import numpy as np
import os
import soundfile as sf
from tqdm import tqdm

np.random.seed(seed=42)

def get_speech_regions(speech_regions_path):
    speech_regions = {}
    for file in os.listdir(speech_regions_path):
        speech_regions[file[:-4]] = np.load(os.path.join(speech_regions_path, file))
    return speech_regions
    
def read_config(config_path):
    config = {}
    class_names = []
    class_probs = []
    with open(config_path) as f:
        lines = [line.rstrip() for line in f.readlines()]
        for line in lines[1:]:
            class_name = line.split()[0]
            class_prob = float(line.split()[-1])
            distrib_name = line.split()[1].split(',')[0]
            distrib_params = list(map(float,line.split()[1].split(',')[1:]))
            class_names.append(class_name)
            class_probs.append(class_prob)
            
            config[class_name] = (distrib_name, distrib_params, class_prob)
    return config, class_names, class_probs
            
    
        

def main():
    orig_base_path = r'/srv/data/raw_data/AMI/amicorpus'
    noises_base_path = r'/srv/data/raw_data/office_noises_for_AMI/simulated_noises'
    speech_regions_path = './speech_regions/'
    config_path = './config'
    out_destination = r'/srv/data/raw_data/AMI/amicorpus_noisy/amicorpus_noised'
    speech_regions = get_speech_regions(speech_regions_path)
    config, class_names, class_probs = read_config(config_path)
    
    noises_list = os.listdir(noises_base_path)
    
    # read session data
    for dir in tqdm(os.listdir(orig_base_path)):
        if 'beamformed' in dir:
            continue
        if dir in os.listdir(out_destination):
            continue
        print('current session is ', dir)
        out_file = open('./noised_corpus_info/{}.txt'.format(dir), 'w')
        lines_to_out_file = []
        
        percent_of_noises = np.random.uniform(0.4, 0.61)
        curr_len_noises = 0
        curr_session = [0 for _ in range(8)]
        
        
        for session_path in os.listdir(os.path.join(orig_base_path, dir, './audio')):
            # read wavs only for Array1
            if session_path.split('.')[1].startswith('Array1'):
                channel_n = int(session_path.split('.')[1].split('-')[-1])
                session_wav, session_sr = sf.read(os.path.join(orig_base_path, dir, './audio', session_path))
                curr_session[channel_n-1] = session_wav
        print('read session is done')
        
       
        curr_session = np.array(curr_session)
        
        if curr_session.ndim != 2:
            continue
        
        
        # calculate mean power of speech during session
        print('start calculate power of speech regions...')
        speech_region_curr_session = np.zeros(curr_session.shape[1])
        for reg in speech_regions[dir]:
            speech_region_curr_session[int(reg[0]*session_sr):int(reg[1]*session_sr)] = 1
        mean_power_session = np.mean(curr_session[:, speech_region_curr_session==1]**2)
        out_file.write('{}, length = {:.2f} s, percent_of_speech = {:.2f}, percent_of_noises = {:.2f}\n'.format(dir,
                                                                                                     curr_session.shape[1]/session_sr, 
                                                                                                     np.mean(speech_region_curr_session), 
                                                                                                     percent_of_noises))
        print('calculation is done!')
        
        
        # do mix with noise until the desired length
        print('start add noises to session {}'.format(dir))
        needed_noise_len = int(curr_session.shape[1]*percent_of_noises)
        
        while curr_len_noises < needed_noise_len:
            noise_class = np.random.choice(class_names, p=class_probs)
            curr_noise_file = np.random.choice([noise for noise in noises_list if noise.startswith(noise_class)])
            
            curr_noise = [0 for _ in range(8)]
            for noise_wav_path in os.listdir(os.path.join(noises_base_path, curr_noise_file)):
                curr_noise_ch = int(noise_wav_path[:-4][-1])
                curr_noise_wav, noise_sr = sf.read(os.path.join(noises_base_path, curr_noise_file, noise_wav_path))
                assert noise_sr == session_sr
                curr_noise[curr_noise_ch] = curr_noise_wav
            curr_noise = np.array(curr_noise)
            if curr_noise.shape[1] < 3*noise_sr:
                curr_noise = np.tile(curr_noise, int(np.ceil(3*noise_sr/curr_noise.shape[1])))
            
            # change noise sound to desired snr
            mean_power_noise = np.mean(curr_noise**2)
            curr_snr = 10*np.log10(mean_power_session/(mean_power_noise+1e-7))
            
            snr_distrib = config[noise_class][0]
            if snr_distrib == 'normal':
                
                mu, std = config[noise_class][1][0], config[noise_class][1][1]
            desired_snr = max(-1,np.random.normal(mu, std))
            
            curr_noise = curr_noise * 10**((curr_snr-desired_snr)/20)
            
            # add noise
            noise_add_start = np.random.randint(0, curr_session.shape[1] - curr_noise.shape[1])
            curr_session[:, noise_add_start:noise_add_start+curr_noise.shape[1]] += curr_noise
            curr_len_noises += curr_noise.shape[1]
            
            # add info about added noise
            lines_to_out_file.append([curr_noise_file, noise_add_start/noise_sr,(noise_add_start+curr_noise.shape[1])/noise_sr,desired_snr])
            
        
        lines_to_out_file = sorted(lines_to_out_file, key=lambda x: x[1])
        for line in lines_to_out_file:
            out_file.write('noise: {}, start: {:.4f} s end: {:.4f} s, duration: {:.4f} s, snr: {:.2f}\n'.format(line[0], line[1], line[2], line[2]-line[1], line[3]))
        # save noised session
        
        out_path = os.path.join(out_destination, dir)
        os.mkdir(out_path)
        for i in range(curr_session.shape[0]):
            out_path_wav = os.path.join(out_path, dir+'.Array1-0{}.wav'.format(i+1))
            sf.write(out_path_wav, curr_session[i,:], session_sr)
        
        out_file.close()
        print('\n\n\n')
  
if __name__ == '__main__':
    main()
                
            
            
            
            
        
