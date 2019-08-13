import pyroomacoustics as pra
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm
np.random.seed(42)

def get_sensor_positions(r=0.1, n=8):

    sensors = []
    ang_step = 360 / n
    for i in range(n):
        x = r * np.cos(np.deg2rad(i * ang_step))
        y = r * np.sin(np.deg2rad(i * ang_step))
        sensors.append([x, y, 0])

    sensors = np.array(sensors).T
    return sensors

def main():

    raw_wav_folder = './raw_noises/'
    noises_dict_path = './noises_config'
    out_wav_folder = './simulated_noises/'
    if not os.path.exists(out_wav_folder):
        os.mkdir(out_wav_folder)
    
    out_info = open('./simulated_files_info.txt', 'a')

    
    
    
    # room configuration, IDIAP room
    length, width, height = (8.5, 3.5, 2.25)
    abs_coef = 0.13
    reflection = 20
    
    # mic array position
    mic_array_center = np.array([2.15, 1.75, 1])
    sensors_position = get_sensor_positions()
    sensors_position = sensors_position + mic_array_center[:, np.newaxis]
    
    # make room
    # room = pra.ShoeBox([length, width, height], absorption=abs_coef, max_order=reflection, fs=16000)
    # room.add_microphone_array(pra.MicrophoneArray(sensors_position, room.fs))
    
    
    # noises dictonary, distance from the array center,  class: (distribution, *params)
    
    noises_dict = {}
    with open(noises_dict_path) as f:
        lines = [line.rstrip() for line in f.readlines()[1: ]]
        
        for line in lines:
            class_name = line.split(',')[0]
            params = line.split(',')[1], line.split(',')[2], line.split(',')[3]
            noises_dict[class_name] = params
    
    for file in tqdm(os.listdir(raw_wav_folder)):
        
        file_class = file.split('_')[0]
        file_params = noises_dict[file_class]
        
        
        # define source position and do simulation
        if file_params[0] == 'normal':
            mu, std = float(file_params[1]), np.sqrt(float(file_params[2]))
            for i in range(5):
                source_dist = np.abs(np.random.normal(mu, std))
                source_phi = np.deg2rad(np.random.uniform(0, 360))
                source_z = min(np.abs(np.random.normal(0, 0.5)), source_dist-0.05)  # distribution for height
                source_theta = np.arccos(source_z/(source_dist+1e-2))
                source_z = source_z + mic_array_center[2]
                source_x = source_dist*np.sin(source_theta)*np.cos(source_phi) + mic_array_center[0]
                source_y = source_dist*np.sin(source_theta)*np.sin(source_phi) + mic_array_center[1]
                
                source_x = max(min(length - 0.05, source_x), 0.05)
                source_y = max(min(width - 0.05, source_y), 0.05)
                source_z = max(min(height - 0.05, source_z), 0.05)
                
                source_position = np.array([source_x, source_y, source_z])
                
                wav, sr = sf.read(raw_wav_folder+file)
                
                room = pra.ShoeBox([length, width, height], absorption=abs_coef, max_order=reflection, fs=sr)
                room.add_microphone_array(pra.MicrophoneArray(sensors_position, room.fs))
                room.add_source(source_position, signal=wav)
                room.image_source_model(use_libroom=True)
                room.simulate()
                
                out_folder_path = out_wav_folder + file[:-4] + '_{}/'.format(i)
                os.mkdir(out_folder_path)
                for j in range(room.mic_array.signals.shape[0]):

                    sf.write(out_folder_path + file[:-4] + '_{}_ch_{}.wav'.format(i,j), room.mic_array.signals[j, :], sr)
                
                out_info.write(file[:-4] + '_{}.wav'.format(i) + ' ' + file + ' ' + str(source_dist) + ' ' + str(source_position) + ' ' + 
                                                            str([np.rad2deg(source_phi), np.rad2deg(source_theta)]) + '\n')
        
        elif file_params[0] == 'uniform':
            
            low, high = float(file_params[1]), float(file_params[2])
            
            for i in range(5):
            
                source_dist = np.random.uniform(low, high)
                source_phi = np.deg2rad(np.random.uniform(0, 360))
                source_z = min(np.abs(np.random.normal(0, 0.5)), source_dist-0.05)  # distribution for height
                source_theta = np.arccos(source_z/(source_dist+1e-2))
                source_z = source_z + mic_array_center[2] 
                source_x = source_dist*np.sin(source_theta)*np.cos(source_phi) + mic_array_center[0]
                source_y = source_dist*np.sin(source_theta)*np.sin(source_phi) + + mic_array_center[1]
                
                source_x = max(min(length - 0.05, source_x), 0.05)
                source_y = max(min(width - 0.05, source_y), 0.05)
                source_z = max(min(height - 0.05, source_z), 0.05)
                
                source_position = np.array([source_x, source_y, source_z])
                
                wav, sr = sf.read(raw_wav_folder+file)
                
                room = pra.ShoeBox([length, width, height], absorption=abs_coef, max_order=reflection, fs=sr)
                room.add_microphone_array(pra.MicrophoneArray(sensors_position, room.fs))
                room.add_source(source_position, signal=wav)
                room.image_source_model(use_libroom=True)
                room.simulate()
                
                out_folder_path = out_wav_folder + file[:-4] + '_{}/'.format(i)
                os.mkdir(out_folder_path)
                for j in range(room.mic_array.signals.shape[0]):

                    sf.write(out_folder_path + file[:-4] + '_{}_ch_{}.wav'.format(i,j), room.mic_array.signals[j, :], sr)
                
                out_info.write(file[:-4] + '_{}.wav'.format(i) + ' ' + file + ' ' + str(source_dist) + ' ' + str(source_position) + ' ' + 
                                                            str([np.rad2deg(source_phi), np.rad2deg(source_theta)]) + '\n')
                                                           
        elif file_params[0] == 'rectangular':
            
            for i in range(5):
                
                case = np.random.choice(4)
                if case == 0 or case == 2:
                    source_x = 0 if case == 0 else length
                    source_y = np.random.uniform(0, width)
                elif case == 1 or case == 3:
                    source_y = width if case == 1 else 0
                    source_x = np.random.uniform(0, length)
                
                source_z = np.random.uniform(0.5, 1.5)
                
                source_dist = (source_x**2 + source_y**2 + (source_z-mic_array_center[2])**2)**0.5
                
                source_theta = np.arccos((source_z-mic_array_center[2])/(source_dist+1e-2))
                source_phi = np.arctan(source_y/(source_x+1e-2))
                
                source_position = np.array([source_x, source_y, source_z])
                
                wav, sr = sf.read(raw_wav_folder+file)
                
                room = pra.ShoeBox([length, width, height], absorption=abs_coef, max_order=reflection, fs=sr)
                room.add_microphone_array(pra.MicrophoneArray(sensors_position, room.fs))
                room.add_source(source_position, signal=wav)
                room.image_source_model(use_libroom=True)
                room.simulate()
                
                out_folder_path = out_wav_folder + file[:-4] + '_{}/'.format(i)
                os.mkdir(out_folder_path)
                for j in range(room.mic_array.signals.shape[0]):

                    sf.write(out_folder_path + file[:-4] + '_{}_ch_{}.wav'.format(i,j), room.mic_array.signals[j, :], sr)
                
                out_info.write(file[:-4] + '_{}.wav'.format(i) + ' ' + file + ' ' + str(source_dist) + ' ' + str(source_position) + ' ' + 
                                                            str([np.rad2deg(source_phi), np.rad2deg(source_theta)]) + '\n')
                

if __name__ == '__main__':

    main()




