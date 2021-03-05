#! /usr/bin/python
import imp
import scipy
import scipy.signal
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from struct import unpack

def maintain_min_shape(working_frame, min_size, frame_array):
    frame_size = len(working_frame)
    if min_size > frame_size:
        min_size = frame_size
        frame_array = frame_array[:, :frame_size]
    elif min_size < frame_size:
        working_frame = working_frame[:min_size]
    return working_frame, min_size, frame_array

def maintain_max_shape(working_frame, max_size, frame_array):
    frame_size = len(working_frame)
    if max_size < frame_size:
        frame_array = [x + ([0] * (frame_size - max_size)) for x in frame_array]
        max_size = frame_size
    elif max_size > frame_size:
        working_frame.extend([0] * (max_size - frame_size))
    return working_frame, max_size, frame_array


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def calculate_rssi(args, raw_frame):
    rssi = sum([x.imag**2 + x.real**2 for x in raw_frame]) / raw_frame.size
    rssi_dbm = 10 * np.log10(rssi) + 30
    if (args.debug):
        print("RSSI: ", rssi)
        print("RSSI in dbm: ", rssi_dbm)
    return rssi_dbm

def skip_frame(index, items_in_frame, parsed_data=None, last_pos=-1):
    if (parsed_data):
        if (last_pos != -1):
            parsed_data.seek(last_pos)
        else:
            Exception("Parsed_data passed but not file position")

    index += items_in_frame
    return index, parsed_data


def crossCor(frame):
    ltf = np.array(
        [complex(-0.0455, -1.0679), complex(0.3528, -0.9865), complex(0.8594, 0.7348), complex(0.1874, 0.2475),
         complex(0.5309, -0.7784), complex(-1.0218, -0.4897), complex(-0.3401, -0.9423), complex(0.8657, -0.2298),
         complex(0.4734, 0.0362), complex(0.0088, -1.0207), complex(-1.2142, -0.4205), complex(0.2172, -0.5195),
         complex(0.5207, -0.1326), complex(-0.1995, 1.4259), complex(1.0583, -0.0363), complex(0.5547, -0.5547),
         complex(0.3277, 0.8728), complex(-0.5077, 0.3488), complex(-1.1650, 0.5789), complex(0.7297, 0.8197),
         complex(0.6173, 0.1253), complex(-0.5353, 0.7214), complex(-0.5011, -0.1935), complex(-0.3110, -1.3392),
         complex(-1.0818, -0.1470), complex(-1.1300, -0.1820), complex(0.6663, -0.6571), complex(-0.0249, 0.4773),
         complex(-0.8155, 1.0218), complex(0.8140, 0.9396), complex(0.1090, 0.8662), complex(-1.3868, -0.0000),
         complex(0.1090, -0.8662), complex(0.8140, -0.9396), complex(-0.8155, -1.0218), complex(-0.0249, -0.4773),
         complex(0.6663, 0.6571), complex(-1.1300, 0.1820), complex(-1.0818, 0.1470), complex(-0.3110, 1.3392),
         complex(-0.5011, 0.1935), complex(-0.5353, -0.7214), complex(0.6173, -0.1253), complex(0.7297, -0.8197),
         complex(-1.1650, -0.5789), complex(-0.5077, -0.3488), complex(0.3277, -0.8728), complex(0.5547, 0.5547),
         complex(1.0583, 0.0363), complex(-0.1995, -1.4259), complex(0.5207, 0.1326), complex(0.2172, 0.5195),
         complex(-1.2142, 0.4205), complex(0.0088, 1.0207), complex(0.4734, -0.0362), complex(0.8657, 0.2298),
         complex(-0.3401, 0.9423), complex(-1.0218, 0.4897), complex(0.5309, 0.7784), complex(0.1874, -0.2475),
         complex(0.8594, -0.7348), complex(0.3528, 0.9865), complex(-0.0455, 1.0679), complex(1.3868, -0.0000)],
        dtype=np.complex64)
    corr = abs(np.correlate(frame, ltf, mode='same')) / ([abs(x) for x in frame])
    return corr


def test_plots(raw_frame, ant2_frame=[], corr_frame=[]):
    fig, ax = plt.subplots(4)
    fig.suptitle('CrossCorrelation, Antenna 1 Mag, Antenna 2 Mag, Freq Plots')
    ax[0].plot(corr_frame, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x), ant2_frame)), 'r')
    ax[3].plot(np.fft.fftshift(np.fft.fft(raw_frame)), 'g')
    plt.show()
    plt.close()

def extract_single_csi(csi_data, csi_index):
    frame_number = unpack('i', csi_data[csi_index:csi_index+4])[0]
    csi_index += 4
    check_delim(csi_data[csi_index])
    csi_index += 1
    size = unpack('i', csi_data[csi_index:csi_index + 4])[0]
    csi_index += 4
    check_delim(csi_data[csi_index])
    csi_index += 1
    csi_bytes = size * 8
    csi_frame = np.array(unpack('f' * size * 2, csi_data[csi_index:csi_index + csi_bytes]), dtype=np.float32)
    csi_frame = list(csi_frame.view(np.complex64))
    csi_index += csi_bytes
    new_line = csi_data[csi_index]
    csi_index += 1
    if (new_line != '\n' or new_line == ''):
        raise("New line character missing in csi file")
    return csi_frame, csi_index, frame_number


def assemble_csi_frame(args, frame_number, csi_data, csi_index):

    while (csi_data[csi_index] != ''):
        csi_frame, csi_index, csi_frame_number = extract_single_csi(csi_data, csi_index)
        if (csi_frame_number == frame_number):
            return csi_frame, csi_index
        elif (csi_frame_number > frame_number):
            raise("Missing CSI Frame")

def trim_wifi(args, raw_frame):
    #0.025 is hand determined value of beginning of array. May need to retune
    abs_array = np.array([abs(x) for x in raw_frame])
    start_index = np.argmax(abs_array > 0.015)
    start_index -= 15
    preamble = np.array([],np.complex64)
    if (start_index < args.trim_len + 15):
        return start_index, preamble
    #20 sample buffer to ensure capture of transient
    for samp in range(args.trim_len):
        preamble = np.append(preamble, raw_frame[start_index + samp])
    return start_index, preamble[1:len(preamble)-1]

def get_AoA(ant1_frame, ant2_frame):

    ang1 = np.mean(np.angle(ant1_frame))
    ang2 = np.mean(np.angle(ant2_frame))
    wv_len = (3.0*10**8)/(2.565*10**9)
    dist = 0.02
    theta = np.arcsin(((ang1-ang2)*wv_len))/(2*np.pi*dist)
    theta_deg = theta * np.pi/180

def save2np(args, training_frames, training_csi, training_pow, training_ant2, training_num, training_time):
    training_frames = np.array(training_frames)
    training_csi = np.array(training_csi)
    training_pow = np.array(training_pow)
    training_ant2 = np.array(training_ant2)
    training_num = np.array(training_num)
    training_time = np.array(training_time)
    if (training_frames.shape[0] != training_csi.shape[0] or training_frames.shape[0] != training_time.shape[0]) or training_frames.shape[0] != training_ant2.shape[0] or training_frames.shape[0] != training_num.shape[0]:
        raise("Samples missing attributes before save")
    final_size = training_frames.shape[0]
    dev_labels = np.full((final_size), args.dev, dtype=np.uint8)
    loc_labels = np.full((final_size,2), list(args.loc), dtype=np.float16)
    rx_labels = np.full((final_size), args.rx, dtype=np.uint8)
    save_path = args.base_path + "ProcData/WiFi" + str(args.loc[0]) + "_" + str(args.loc[1]) + "_" + str(args.dev) + "_" + str(args.rx)
    np.savez(save_path, rawTrain=training_frames, ant2Train=training_ant2, csiTrain=training_csi, powTrain=training_pow, frameNums=training_num, frameTimes=training_time,
             dev_labels=dev_labels, loc_labels=loc_labels, rx_labels=rx_labels)

    #print("Number of Frames Captured:", final_size)
    #print("Done")

def check_delim(buffer):
    if (buffer != '|'):
        raise("Deliminator not in correct position")

def unpack_frame(pd, index):
    # mac_addr = pd.read(17)
    # start = pd.tell()
    # while (pd.tell() - start < 17):
    #     mac_char = pd.read(1)
    # check_delim(pd)
    frame_number = unpack('i', pd[index:index+4])[0]
    index += 4
    check_delim(pd[index])
    index += 1
    seq_num = unpack('i', pd[index:index + 4])[0]
    index += 4
    check_delim(pd[index])
    index += 1
    time_ns = unpack('q', pd[index:index + 8])[0]
    index += 8
    check_delim(pd[index])
    index += 1
    raw_size = unpack('i', pd[index:index + 4])[0]
    index += 4
    check_delim(pd[index])
    index += 1
    raw_ant2_size = unpack('i', pd[index:index + 4])[0]
    index += 4
    check_delim(pd[index])
    index += 1
    raw_size_bytes = raw_size * 8
    raw_ant2_bytes = raw_ant2_size * 8
    raw_frame = np.array(unpack('f' * raw_size * 2, pd[index:index + raw_size_bytes]), dtype=np.float32)
    raw_frame = list(raw_frame.view(np.complex64))
    index += raw_size_bytes
    check_delim(pd[index])
    index += 1
    ant2_frame = np.array(unpack('f' * raw_ant2_size * 2, pd[index:index + raw_ant2_bytes]), dtype=np.float32)
    ant2_frame = list(ant2_frame.view(np.complex64))
    index += raw_ant2_bytes
    size_diff = len(ant2_frame) - len(raw_frame)
    if (size_diff > 0):
        ant2_frame = ant2_frame[:len(ant2_frame) - abs(size_diff)]
    elif (size_diff < 0):
        raw_frame  = raw_frame[:len(raw_frame) - abs(size_diff)]
    new_line = pd[index]
    index += 1
    if (new_line != '\n' or new_line == ''):
        raise("New line character missing in parsed_data file")
    return frame_number, seq_num, time_ns, raw_frame, ant2_frame, index

def wifi_main(args):
    # Load Files
    start_num = 0
    # curr_min_size = args.trim_len
    curr_max_size = 0
    curr_max_csi_size = 0
    curr_max_ant2_size = 0
    training_raw_frames = []
    training_ant2_frames = []
    training_csi_frames = []
    training_rssi = []
    training_seq = []
    training_time = []
    csi_index = 0;
    csi_file = open(args.base_path + "RawData/csi_data" + args.fileName).read()

    with open(args.base_path + "RawData/parsed_data" + args.fileName, "rb") as pd:
        # starting_pos = pd.tell()
        # pd.read()
        # ending_pos = pd.tell()
        # pd.seek(starting_pos)
        data = pd.read()
        index = 0
        while (index != pd.tell()):
            if (len(training_raw_frames) == args.num_save):
                break
            frame_number, seq_num, time_ns, raw_frame, ant2_frame, index = unpack_frame(data, index)

            if (start_num < 50):
                start_num += 1
                continue
            elif (len(raw_frame) == 0 or len(ant2_frame) == 0):
                continue
            elif (len(raw_frame) > 10000):
                continue
            """elif (args.target_mac != mac_addr and args.target_mac):
                continue"""

            # Calls cross correlation function, trims frame to preamble, and ensures the frames start is near its max corr
            if (args.debug):
                corr_frame = crossCor(raw_frame)
                corr_frame = np.delete(corr_frame, np.where(corr_frame == np.inf))
                # Also plots the test_plots to view the frame
                if (args.plot_frames):
                    test_plots(raw_frame, ant2_frame, corr_frame)
            elif (args.plot_frames):
                test_plots(raw_frame, ant2_frame)

            # Save raw frame, csi (if desired), and rssi (if desired)
            raw_frame, curr_max_size, training_raw_frames = maintain_max_shape(raw_frame, curr_max_size, training_raw_frames)
            training_raw_frames.append(raw_frame)
            ant2_frame, curr_max_ant2_size, training_ant2_frames = maintain_max_shape(ant2_frame, curr_max_ant2_size, training_ant2_frames)
            training_ant2_frames.append(ant2_frame)

            if (args.got_csi):
                csi_frame = assemble_csi_frame(args, frame_number, csi_file, csi_index)
                csi_frame, curr_max_csi_size, training_csi_frames = maintain_max_shape(csi_frame, curr_max_csi_size, training_csi_frames)
                training_csi_frames.append(csi_frame)
                if (args.plot_frames and args.debug):
                    test_plots(csi_frame)
            """if (args.got_power):
                rssi = calculate_rssi(args, raw_frame)
                training_rssi = np.vstack([training_rssi, rssi])"""
            if (args.got_seq):
                training_seq.append(seq_num)
            if (args.got_time):
                training_time.append(time_ns)
            #print("Frame", len(training_raw_frames), " captured")

    save2np(args, training_raw_frames, training_csi_frames, training_rssi, training_ant2_frames, training_seq, training_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes 5 files (file sink, file sink header, parse data, and '
                                                 'CSI data and organizes them into a text file where each lline corresponds to a single frame',
                                     version=1.0)
    parser.add_argument('-wifi', action='store_true', default=False, dest='parse_wifi', help='Parses WiFi')
    parser.add_argument('-file', action='store', default='', dest='fileName',
                        help='File suffix at the end of each file')
    parser.add_argument('-base_path', action='store', default="/home/nij/GNU/", dest='base_path',
                        help='Path to all files')
    parser.add_argument('-p', action='store_true', default=False, dest='plot_frames',
                        help='Plots frames in matplotlib to appear on screen')
    parser.add_argument('-n', action='store', type=int, default=1500, dest='num_save',
                        help='The upper bound of how many frames will be saved to file')
    parser.add_argument('-d', action='store_true', default=False, dest='debug', help='Turns on debug mode')
    parser.add_argument('-no_csi', action='store_false', default=True, dest='got_csi', help='No CSI data for set')
    parser.add_argument('-no_power', action='store_false', default=True, dest='got_power', help='No RSSI data for set')
    parser.add_argument('-no_seq', action='store_false', default=True, dest='got_seq', help='No sequence numbers data for set')
    parser.add_argument('-no_time', action='store_false', default=True, dest='got_time', help='No timestamp data for set')
    parser.add_argument('-mac', action='store', default="", dest='target_mac',
                        help='MAC Address to specify the only device to be saved')
    parser.add_argument('-dev', action='store', type=int, default=None, dest='dev',
                        help='Class number of the transmitting device')
    parser.add_argument('-loc', nargs='+', action='store', type=int, default=(None, None), dest='loc',
                        help='Location point of the transmitting device')
    parser.add_argument('-rx', action='store', type=int, default=None, dest='rx',
                        help='Class number of the receiving device')
    parser.add_argument('-tl', action='store', type=int, default=350, dest='trim_len',
                        help='Number of samples to trim from beginning of frame (Only applicable to WiFi)')
    args = parser.parse_args()

    wifi_main(args)
