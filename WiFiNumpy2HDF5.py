#! /usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import argparse
from os import listdir
from os.path import isfile, join
import timeit, h5py
from math import log10
from statistics import mean
import csv

class SDR:
    def __init__(self, file):
        self.ant1 = file['rawTrain']
        self.ant2 = file['ant2Train']
        self.loc = file['loc_labels']
        self.loc = self.loc.reshape((self.loc.shape[0]//2, 2)).tolist()
        self.dev = file['dev_labels'].tolist()
        self.rx = file['rx_labels'].tolist()
        self.csi = file['csiTrain'].tolist()
        self.rssi = file['powTrain'].tolist()
        self.num = file['frameNums'].tolist()
        self.snr = 0
        self.time = file['frameTimes'].tolist()

    def process_frames(self, args):
        trimmed_ant1_frames = []
        trimmed_ant2_frames = []
        rssi = []
        snr = []
        delete_list = []
        for index, frames in enumerate(zip(self.ant1, self.ant2)):
            raw_frame = frames[0].tolist()
            ant2_frame = frames[1].tolist()
            corr_frame = []
            if args.debug:
                corr_frame = crossCor(raw_frame)
            if args.plot_frames:
                test_plots(raw_frame, ant2_frame, corr_frame)
            raw_start, raw_preamble, noise_frame = trim_wifi(args, raw_frame)
            ant2_start, ant2_preamble, _ = trim_wifi(args, ant2_frame)
            if raw_preamble.size != 0 and ant2_preamble.size != 0:
                if args.plot_frames:
                    test_plots(raw_preamble, ant2_preamble, corr_frame[:raw_start + 200])
                trimmed_ant1_frames.append(raw_preamble)
                trimmed_ant2_frames.append(ant2_preamble)
                rssi_new = calculate_rssi(args, raw_preamble, True)
                rssi_snr = np.multiply(raw_preamble, np.conjugate(raw_preamble))
                rssi_snr = np.mean(rssi_snr)
                np_snr = np.multiply(noise_frame, np.conjugate(noise_frame))
                np_snr = np.mean(np_snr)

                rssi.append(rssi_new)
                snr.append((rssi_snr/np_snr) - 1)
            else:
                delete_list.append(index)

        for index in sorted(delete_list, reverse=True):
            del self.loc[index]
            del self.dev[index]
            del self.rx[index]
            del self.csi[index]
            del self.time[index]
            del self.num[index]
        self.ant1 = trimmed_ant1_frames
        self.ant2 = trimmed_ant2_frames
        self.rssi = rssi
        self.snr = 10*log10(np.mean(snr))

def maintain_min_shape(working_frames, min_size, frame_array):
    frame_size = len(working_frames[0])
    if min_size > frame_size:
        min_size = frame_size
        frame_array = frame_array[:, :frame_size]
    elif min_size < frame_size:
        working_frames = working_frames[:, :min_size]
    return working_frames, min_size, frame_array

def maintain_max_shape(working_frames, max_size, frame_array):
    frame_size = len(working_frames[0])
    if max_size < frame_size:
        frame_array = [x + ([0] * (frame_size - max_size)) for x in frame_array]
        max_size = frame_size
    elif max_size > frame_size:
        working_frames = [x + ([0] * (max_size - frame_size)) for x in working_frames]
    return working_frames, max_size, frame_array

def calculate_rssi(args, raw_frame, dbm):
    rssi = sum([x.imag**2 + x.real**2 for x in raw_frame]) / raw_frame.size
    if dbm:
        rssi_dbm = 10 * np.log10(rssi) + 30
    if (args.debug):
        print("RSSI: ", rssi)
        if dbm:
            print("RSSI in dbm: ", rssi_dbm)
    if dbm:
        return rssi_dbm
    else:
        return rssi

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

def pool_seq(sdr_num):
    start_bool = [i > j for i, j in zip(sdr_num, sdr_num[1:])]
    startovers = start_bool.count(True)
    last_number = sdr_num[-1]
    total_sent = startovers * 4096 + last_number
    return total_sent


def test_plots(raw_frame, ant2_frame=[], corr_frame=[]):
    fig, ax = plt.subplots(4)
    fig.suptitle('CrossCorrelation, Antenna 1 Mag, Antenna 2 Mag, Freq Plots')
    ax[0].plot(corr_frame, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x), ant2_frame)), 'r')
    ax[3].plot(np.fft.fftshift(np.fft.fft(raw_frame)), 'g')
    plt.show()

def trim_wifi(args, raw_frame):
    #0.025 is hand determined value of beginning of array. May need to retune
    abs_array = np.array([abs(x) for x in raw_frame])
    start_index = np.argmax(abs_array > 0.015)
    start_index -= 15
    preamble = np.array([],np.complex64)
    noise_frame = np.array([],np.complex64)
    if (start_index < args.trim_len + 15):
        return start_index, preamble, []
    #20 sample buffer to ensure capture of transient
    for samp in range(args.trim_len):
        preamble = np.append(preamble, raw_frame[start_index + samp])
    for samp in range(start_index):
        noise_frame = np.append(noise_frame, raw_frame[samp])
    return start_index, preamble[1:len(preamble)-1], noise_frame

def writeCSV(args, csv_rows):
    fields = ['X-Coordinate', 'Y-Coordinate', 'Phone Label', 'SDR Label', 'Total Frames Sent', 'SINR']
    filename = args.base_path + "nij_collection_stats_part2.csv"

    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(csv_rows)

def wifi_main(args):
    # Load all Files from folder
    sdr1_files = [f for f in listdir(args.base_path) if isfile(join(args.base_path, f)) and f.endswith("1.npz")]
    sdr2_files = [f for f in listdir(args.base_path) if isfile(join(args.base_path, f)) and f.endswith("2.npz")]
    sdr3_files = [f for f in listdir(args.base_path) if isfile(join(args.base_path, f)) and f.endswith("3.npz")]
    sdr5_files = [f for f in listdir(args.base_path) if isfile(join(args.base_path, f)) and f.endswith("5.npz")]

    #Sort files and ensure they match/nothing is missing
    sdr1_files.sort()
    sdr2_files.sort()
    sdr3_files.sort()
    sdr5_files.sort()
    sdr1_suffix = [sdr.replace("WiFiSDR", '')[:-5] for sdr in sdr1_files]
    sdr2_suffix = [sdr.replace("WiFiSDR", '')[:-5] for sdr in sdr2_files]
    sdr3_suffix = [sdr.replace("WiFiSDR", '')[:-5] for sdr in sdr3_files]
    sdr5_suffix = [sdr.replace("WiFiSDR", '')[:-5] for sdr in sdr5_files]
    if (sdr1_suffix != sdr2_suffix or sdr1_suffix != sdr3_suffix or sdr1_suffix != sdr5_suffix):
        raise(Exception("Number of files from each SDR not equal"))
    out_file = h5py.File("WiFiClass1train.hdf5", 'w')
    total_ant1 = out_file.create_dataset('ant1Train', shape=(0,348), dtype=np.complex64, chunks=True, compression='lzf', maxshape=(None, None))
    total_ant2 = out_file.create_dataset('ant2Train', shape=(0,348), dtype=np.complex64, chunks=True, compression='lzf', maxshape=(None, None))
    total_csi = out_file.create_dataset('csiTrain', shape=(0,10240), dtype=np.complex64, chunks=True, compression='lzf', maxshape=(None, None))
    total_rssi = out_file.create_dataset('powTrain', shape=(0,), dtype=np.float64, chunks=True, compression='lzf', maxshape=(None,))
    total_dev = out_file.create_dataset('dev_labels', shape=(0,), dtype=np.uint8, chunks=True, compression='lzf', maxshape=(None,))
    total_loc = out_file.create_dataset('loc_labels', shape=(0,2), dtype=np.float16, chunks=True, compression='lzf', maxshape=(None, 2))
    total_rx = out_file.create_dataset('rx_labels', shape=(0,), dtype=np.uint8, chunks=True, compression='lzf', maxshape=(None,))
    total_time = out_file.create_dataset('timeTrain', shape=(0,), dtype=np.float64, chunks=True, compression='lzf', maxshape=(None,))
    total_num = out_file.create_dataset('numTrain', shape=(0,), dtype=np.uint16, chunks=True, compression='lzf', maxshape=(None,))
    csv_rows = []
    max_csi_size = 10240
    for sdr1_file, sdr2_file, sdr3_file, sdr5_file in zip(sdr1_files, sdr2_files, sdr3_files, sdr5_files):
        start_time = timeit.default_timer()
        sdr1_np = np.load(join(args.base_path, sdr1_file), mmap_mode='r')
        sdr2_np = np.load(join(args.base_path, sdr2_file), mmap_mode='r')
        sdr3_np = np.load(join(args.base_path, sdr3_file), mmap_mode='r')
        sdr5_np = np.load(join(args.base_path, sdr5_file), mmap_mode='r')

        sdrs = [SDR(sdr1_np), SDR(sdr2_np), SDR(sdr3_np), SDR(sdr5_np)]
        for sdr in sdrs:
            total_sent = pool_seq(sdr.num)
            if sdr.ant1.shape[0] > 1500:
                sdr.ant1 = sdr.ant1[:1500]
                sdr.ant2 = sdr.ant2[:1500]
                sdr.csi = sdr.csi[:1500]
                sdr.dev = sdr.dev[:1500]
                sdr.rx = sdr.rx[:1500]
                sdr.loc = sdr.loc[:1500]
                sdr.time = sdr.time[:1500]
                sdr.num = sdr.num[:1500]
            sdr.process_frames(args)
            csv_rows.append({'X-Coordinate':sdr.loc[0][0], 'Y-Coordinate':sdr.loc[0][1], 'Phone Label':sdr.dev[0], 'SDR Label':sdr.rx[0], 'Total Frames Sent': total_sent, 'SINR': sdr.snr})
            total_ant1.resize((total_ant1.shape[0] + len(sdr.ant1)), axis=0)
            total_ant1[-len(sdr.ant1):] = sdr.ant1
            total_ant2.resize((total_ant2.shape[0] + len(sdr.ant2)), axis=0)
            total_ant2[-len(sdr.ant2):] = sdr.ant2
            total_rssi.resize((total_rssi.shape[0] + len(sdr.rssi)), axis=0)
            total_rssi[-len(sdr.rssi):] = sdr.rssi
            total_dev.resize((total_dev.shape[0] + len(sdr.dev)), axis=0)
            total_dev[-len(sdr.dev):] = sdr.dev
            total_loc.resize((total_loc.shape[0] + len(sdr.loc)), axis=0)
            total_loc[-len(sdr.loc):] = sdr.loc
            total_rx.resize((total_rx.shape[0] + len(sdr.rx)), axis=0)
            total_rx[-len(sdr.rx):] = sdr.rx
            total_time.resize((total_time.shape[0] + len(sdr.time)), axis=0)
            total_time[-len(sdr.time):] = sdr.time
            total_num.resize((total_num.shape[0] + len(sdr.num)), axis=0)
            total_num[-len(sdr.num):] = sdr.num
            frame_size = len(sdr.csi[1])
            if max_csi_size < frame_size:
                total_csi.resize(frame_size, axis=1)
                max_csi_size = frame_size
            elif max_csi_size > frame_size:
                sdr.csi = [x + ([0] * (max_csi_size - frame_size)) for x in sdr.csi]
            total_csi.resize((total_csi.shape[0] + len(sdr.csi)), axis=0)
            total_csi[-len(sdr.csi):] = sdr.csi
        elapsed_time = timeit.default_timer() - start_time
        print("File " + sdr1_file[:-6] + " completed in " + str(elapsed_time) + " seconds")
    
    writeCSV(args, csv_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes 5 files (file sink, file sink header, parse data, and '
                                                 'CSI data and organizes them into a text file where each lline corresponds to a single frame')
    parser.add_argument('-base_path', action='store', default="/home/nij/GNU/RawData/", dest='base_path',
                        help='Path to all files')
    parser.add_argument('-p', action='store_true', default=False, dest='plot_frames',
                        help='Plots frames in matplotlib to appear on screen')
    parser.add_argument('-n', action='store', type=int, default=1500, dest='num_save',
                        help='The upper bound of how many frames will be saved to file')
    parser.add_argument('-d', action='store_true', default=False, dest='debug',
                        help='Debugger mode')
    parser.add_argument('-tl', action='store', type=int, default=350, dest='trim_len',
                        help='Number of samples to trim from beginning of frame (Only applicable to WiFi)')
    args = parser.parse_args()

    wifi_main(args)
