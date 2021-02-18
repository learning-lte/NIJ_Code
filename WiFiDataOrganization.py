#! /usr/bin/python
import imp
import scipy
import scipy.signal
import numpy as np
import sys
import uuid
import matplotlib.pyplot as plt
import copy
import argparse

sys.path.insert(0, '/home/nick/GNU/main/lib/python2.7/dist-packages/')
sys.path.insert(0, '/home/nick/GNU/main/lib/python2.7/dist-packages/pmt/')
sys.path.insert(0, '/home/nick/GNU/main/lib/')
parseHeader = imp.load_source('gr_read_file_metadata_nick', '/home/nick/GNU/main/bin/gr_read_file_metadata_nick')


def maintain_shape(working_frame, min_size, frame_array):
    frame_size = len(working_frame)
    if min_size > frame_size:
        min_size = frame_size
        frame_array = frame_array[:, :frame_size]
    elif min_size < frame_size:
        working_frame = working_frame[:min_size]
    return working_frame, min_size, frame_array


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
    """if max_needed:
        highCorr = [(i, j) for (i, j) in enumerate(corr) if (5 <= j <= 12)]
        startFrame = None
        if (highCorr):
            while (highCorr):
                if (highCorr[0][0] < 320):
                    del highCorr[0]
                else:
                    break
            if (highCorr):
                if max(list(map(lambda x: abs(x), frame))) < 1:
                    startFrame = (highCorr[0][0] - 250)
        return startFrame"""
    return corr


def test_plots(raw_frame, corr_frame=[]):
    fig, ax = plt.subplots(4)
    fig.suptitle('CrossCorrelation, Magnitude, Power, Freq Plots')
    ax[0].plot(corr_frame, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x) ** 2, raw_frame)), 'r')
    ax[3].plot(np.fft.fftshift(np.fft.fft(raw_frame)), 'g')
    plt.show()
    plt.close()


def assemble_csi_frame(args, frame_number):
    csi_data = open(args.base_path + "csi_data" + args.fileName + ".txt").read()
    linestart = csi_data.find(str(frame_number))
    if (linestart < 0):
        Exception("Could not find CSI for associated frame!")
    csistart = csi_data.find("|", linestart, len(csi_data)) + 1
    csiend = csi_data.find("\n", csistart, len(csi_data))
    csi_frame = csi_data[csistart:csiend]
    csi_frame = csi_frame.replace(" ", "")
    csi_frame = csi_frame.replace("(", "")
    csi_frame = csi_frame.replace(")", "")
    csi_frame = csi_frame.split(",")[:-1]
    csi_frame = np.array([complex(x) for x in csi_frame], np.complex64)
    return csi_frame

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

def save2np(args, training_frames, training_csi, training_pow, training_ant2=np.array([]), training_num=np.array([])):
    if (training_frames.shape[0] != training_csi.shape[0] or training_frames.shape[0] != training_pow.shape[0]) or training_frames.shape[0] != training_ant2.shape[0] or training_frames.shape[0] != training_num.shape[0]:
        Exception("Samples missing attributes before save")
    training_frames = training_frames[1:,:]
    training_csi = training_csi[1:,:]
    training_pow = training_pow[1:]
    training_ant2 = training_ant2[1:,:]
    training_num = training_num[1:]
    final_size = training_frames.shape[0]
    dev_labels = np.full((final_size), args.dev, dtype=np.uint8)
    loc_labels = np.full((final_size,2), list(args.loc), dtype=np.float16)
    rx_labels = np.full((final_size), args.rx, dtype=np.uint8)
    if (args.parse_lte):
        save_path = args.base_path + "LTE" + str(args.dev) + "." + str(args.loc[0]) + "." + str(args.loc[1]) + "." + str(args.rx)
    elif (args.parse_wifi):
        save_path = args.base_path + "WiFi" + str(args.loc[0]) + "." + str(args.loc[1]) + "." + str(args.dev) + "." + str(args.rx)
    np.savez(save_path, rawTrain=training_frames, ant2Train=training_ant2, csiTrain=training_csi, powTrain=training_pow, frameNums=training_num,
             dev_labels=dev_labels, loc_labels=loc_labels, rx_labels=rx_labels)

    print("Number of Frames Captured:", final_size)
    print("Done")


def wifi_main(args):
    # Load Files
    raw_data = scipy.fromfile(open(args.base_path + "file_sink" + args.fileName), dtype=scipy.complex64)
    parsed_data = open(args.base_path + "parsed_data" + args.fileName + ".txt", "r")

    # Returns a 2D array containing raw data header info sorted by frame
    # Filters deletes NULL frames
    # Size in Items = Size in Complex Numbers
    # Header Format: HEADER LENGTH, SIZE IN Items, Frame Counter

    header_data = parseHeader.main(args.base_path + "file_sink" + args.fileName + ".hdr", True)
    header_data = list(filter(None, header_data))

    # Sort raw data into frames based on header info
    # Combines parsed data, raw data and assigns frames ID if the raw frame wasnt skipped
    # Parsed Data Format: Transmitter MAC, Frame Number (Counter)
    # CSI Data Format: Frame Number(Counter)|[CSI DATA]
    # Final Format: MAC ADDRESS, FRAME ID, FRAME LENGTH (in items), [RAW DATA], [CSI DATA]
    raw_frame = np.array([], np.complex64)
    index = 0
    start_num = 0
    curr_min_size = args.trim_len
    curr_min_csi_size = 1000
    training_raw_frames = np.empty((1, curr_min_size), np.complex64)
    training_csi_frames = np.empty((1, curr_min_size), np.complex64)
    training_rssi = np.empty((1,1), np.float)

    for header in header_data:

        if (training_raw_frames.shape[0] == args.num_save):
            break
        # If not at end of parsed_data file, reads in next line
        # Frame_number is the number of the next frame that was decoded successfully a.k.a we have its MAC address
        # raw_frame_number is retrieved from the raw frames metadata and the number associated with each raw_frame
        last_pos = parsed_data.tell()
        current_line = parsed_data.read()
        raw_frame_number = int(header[2])

        if (header[1] != 0):
            if (current_line != ""):
                parsed_data.seek(last_pos)
                mac_addr, frame_number = parsed_data.readline().split(",")
                frame_number = int(frame_number.strip("\n"))
            else:
                break

            # If the next frame number is the same as this frame number, skip the raw frame and the frame numbers
            # Else rewind back so the next iteration uses the next frame number
            nxt_last_pos = parsed_data.tell()
            temp = parsed_data.readline()
            if (temp != ""):
                nxt_frame_number = temp.split(",")[1]
                nxt_frame_number = int(nxt_frame_number.strip("\n"))
                if (nxt_frame_number == frame_number):
                    index = skip_frame(index, header[1])[0]
                    continue
                else:
                    parsed_data.seek(nxt_last_pos)
            else:
                parsed_data.seek(nxt_last_pos)
            """Frame Number is the number associated with the frame in the parsed data text file. Raw Frame Number is the number
            saved in the metadata file associated with that raw frame. Both should always be increasing (although Frame Number can
            have duplicates). This if statement checks to see:
                1. if frame number is less than raw_frame_number (Should never happen), raise exception
                2. if frame number is greater than raw_frame_number (most common), then keep frame number and skip raw_frame
                3. Skip 1st 10 frames captured correctly and if any frames come from zeroed MAC addresses (potential virtual machines or spoofs)
                4. If this frame does not have the target MAC address, skip it
                5. Proceed with saving successfully captured frame"""

            if (frame_number < raw_frame_number):
                raise Exception("Recording Error!!!!")
            elif (frame_number > raw_frame_number):
                index, parsed_data = skip_frame(index, header[1], parsed_data, last_pos)
            elif (mac_addr == "00:0:0:0:0:0" or header[1] <= 10000 or start_num < 10):
                start_num += 1
                index = skip_frame(index, header[1])[0]
            elif (args.target_mac != mac_addr and args.target_mac):
                index = skip_frame(index, header[1])[0]
                start_num += 1
            else:

                # Reads in raw frame and increments index for binary file
                for each_item in range(header[1]):
                    raw_frame = np.append(raw_frame, np.complex64(raw_data[index + each_item]))
                index += header[1]

                # Calls cross correlation function, trims frame to preamble, and ensures the frames start is near its max corr
                corr_frame = crossCor(raw_frame)
                corr_frame = np.delete(corr_frame, np.where(corr_frame == np.inf))
                max_corr = np.argmax(corr_frame)

                if (args.plot_frames and args.debug):
                    test_plots(raw_frame, corr_frame)

                start_ind, raw_frame = trim_wifi(args, raw_frame)

                if (abs(start_ind - max_corr) > 8000 or start_ind < 500 or np.count_nonzero(abs(raw_frame)==0) > 5):
                    continue

                # Also plots the test_plots to view the frame
                if (args.plot_frames):
                    test_plots(raw_frame, corr_frame)
                # Save raw frame, csi (if desired), and rssi (if desired)
                raw_frame, curr_min_size, training_raw_frames = maintain_shape(raw_frame, curr_min_size, training_raw_frames)
                training_raw_frames = np.vstack([training_raw_frames, raw_frame])
                if (args.got_csi):
                    csi_frame = assemble_csi_frame(args, frame_number)
                    csi_frame, curr_min_csi_size, training_csi_frames = maintain_shape(csi_frame, curr_min_csi_size, training_csi_frames)
                    training_csi_frames = np.vstack([training_csi_frames, csi_frame])
                if(args.got_power):
                    rssi = calculate_rssi(args, raw_frame)
                    training_rssi = np.vstack([training_rssi, rssi])

        raw_frame = np.array([], np.complex64)

    # Checks to make sure no extra frames captured (Never been a problem before)
    parsed_data.close()
    if (file_len(args.base_path + "parsed_data" + args.fileName + ".txt") < training_raw_frames.shape[0]):
        raise Exception("Captured too many frames.")
    save2np(args, training_raw_frames, training_csi_frames, training_rssi)


def lte_main(args):

    raw_data = scipy.fromfile(open(args.base_path + "raw_data" + args.fileName + ".txt"), dtype = scipy.complex64)
    raw_delim = 999999+0j
    ant2_delim = 999999.1+0j
    csi_delim = 999999.2+0j
    rsrp_delim = 999999.3+0j
    tti_delim = 999999.4 + 0j
    frame_count = np.count_nonzero((raw_data == raw_delim))
    ant2_count = np.count_nonzero((raw_data == ant2_delim))
    csi_count = np.count_nonzero((raw_data == csi_delim))
    rsrp_count = np.count_nonzero((raw_data == rsrp_delim))
    tti_count = np.count_nonzero((raw_data == tti_delim))

    if (frame_count != csi_count or frame_count != rsrp_count or frame_count != ant2_count or frame_count != tti_count):
        Exception("Error: Uneven number of Frames, CSI, and RSRP!")

    skip_samp1_ind = np.where(raw_data==tti_delim)[0][0] + 1
    raw_data = raw_data[skip_samp1_ind:]
    MIN_DESIRED_CSI_LEN = 100
    working_frame = []
    curr_min_size = 500
    curr_min_ant2_size = 500
    curr_min_csi_size = 1000
    training_raw_frames = np.empty((1, curr_min_size), np.complex64)
    training_csi_frames = np.empty((1, curr_min_csi_size), np.complex64)
    training_ant2_frames = np.empty((1, curr_min_size), np.complex64)
    training_rsrp = np.empty((1,1), np.float)
    training_tti = np.empty((1, 1), np.uint32)
    delete_frame_flag = False

    for sample in raw_data:
        if (training_tti.shape == args.num_save):
            break
        if sample == np.complex64(raw_delim):
            if (args.debug):
                print("Length of Pre Raw Frame: ", len(working_frame))
                working_frame = list(filter(lambda samp: samp != 0, working_frame))
                print("Length of Post Raw Frame: ", len(working_frame))
            if (args.plot_frames or training_raw_frames.shape[0] > 8000):
                test_plots(working_frame, np.correlate(working_frame, working_frame, 'same'))

            working_frame, curr_min_size, training_raw_frames = maintain_shape(working_frame, curr_min_size, training_raw_frames)
            raw_frame = np.array(working_frame)
            raw_angle = np.angle(raw_frame, deg=True)
            training_raw_frames = np.vstack([training_raw_frames, raw_frame])
            working_frame = []
        elif sample == np.complex64(ant2_delim):
            if (args.debug):
                print("Length of Pre Raw Frame: ", len(working_frame))
                working_frame = list(filter(lambda samp: samp != 0, working_frame))
                print("Length of Post Raw Frame: ", len(working_frame))
            if (args.plot_frames):
                test_plots(working_frame, np.correlate(working_frame, working_frame, 'same'))

            working_frame, curr_min_ant2_size, training_ant2_frames = maintain_shape(working_frame, curr_min_ant2_size, training_ant2_frames)
            ant2_frame = np.array(working_frame)
            training_ant2_frames = np.vstack([training_ant2_frames, ant2_frame])
            #get_AoA(raw_frame, ant2_frame)
            working_frame = []
        elif sample == np.complex64(csi_delim):
            if (args.debug):
                print("Length of CSI Frame: ", len(working_frame))
                if (args.plot_frames):
                    test_plots(working_frame)
            if (len(working_frame) < MIN_DESIRED_CSI_LEN):
                delete_frame_flag = True
                working_frame = []
                continue

            working_frame = working_frame[12:]
            working_frame, curr_min_csi_size, training_csi_frames = maintain_shape(working_frame, curr_min_csi_size, training_csi_frames)
            csi_frame = np.array(working_frame)
            training_csi_frames = np.vstack([training_csi_frames, csi_frame])
            working_frame = []
        elif sample == np.complex64(rsrp_delim):
            if (len(working_frame) != 1):
                Exception("Extra Elements captured with RSRP")
            rsrp = np.array(float(working_frame[0]))
            if (args.debug):
                print("RSRP in dbm:", float(working_frame[0]))

            training_rsrp = np.vstack((training_rsrp, rsrp))
            working_frame = []

        elif sample == np.complex64(tti_delim):
            if (len(working_frame) != 1):
                Exception("Extra Elements captured with TTI")
            tti = np.array(int(float(working_frame[0])))
            if (args.debug):
                print("TTI:", int(float(working_frame[0])))
            if (delete_frame_flag):
                training_raw_frames = training_raw_frames[:-1]
                training_ant2_frames = training_ant2_frames[:-1]
                training_rsrp = training_rsrp[:-1]
                delete_frame_flag = False
                working_frame = []
                if (args.debug):
                    print("Deleted frame")
                continue

            training_tti = np.vstack((training_tti, tti))
            working_frame = []

        else:
            working_frame.append(sample)

    save2np(args, training_raw_frames, training_csi_frames, training_rsrp, training_ant2_frames, training_tti)

def get_AoA(ant1_frame, ant2_frame):

    ang1 = np.mean(np.angle(ant1_frame))
    ang2 = np.mean(np.angle(ant2_frame))
    wv_len = (3.0*10**8)/(2.565*10**9)
    dist = 0.02
    theta = np.arcsin(((ang1-ang2)*wv_len))/(2*np.pi*dist)
    theta_deg = theta * np.pi/180


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes 5 files (file sink, file sink header, parse data, and '
                                                 'CSI data and organizes them into a text file where each lline corresponds to a single frame',
                                     version=1.0)
    parser.add_argument('-lte', action='store_true', default=True, dest='parse_lte', help='Parses LTE files')
    parser.add_argument('-wifi', action='store_true', default=False, dest='parse_wifi', help='Parses WiFi')
    parser.add_argument('-file', action='store', default='', dest='fileName',
                        help='File suffix at the end of each file')
    parser.add_argument('-base_path', action='store', default="/home/nick/srsLTE/", dest='base_path',
                        help='Path to all files')
    parser.add_argument('-p', action='store_true', default=False, dest='plot_frames',
                        help='Plots frames in matplotlib to appear on screen')
    parser.add_argument('-n', action='store', type=int, default=float('inf'), dest='num_save',
                        help='The upper bound of how many frames will be saved to file')
    parser.add_argument('-d', action='store_true', default=False, dest='debug', help='Turns on debug mode')
    parser.add_argument('-no_csi', action='store_false', default=True, dest='got_csi', help='No CSI data for set')
    parser.add_argument('-no_power', action='store_false', default=True, dest='got_power', help='No RSSI data for set')
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

    if (args.parse_wifi):
        wifi_main(args)
    elif (args.parse_lte):
        lte_main(args)
    else:
        print("Choose to either parse wifi or lte")
