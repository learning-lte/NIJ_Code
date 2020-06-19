#! /usr/bin/python
import imp
import scipy
import scipy.signal
import numpy
import sys
import uuid
import matplotlib.pyplot as plt
import copy
import argparse

sys.path.insert(0, '/home/nij/GNU/main/lib/python2.7/dist-packages/')
sys.path.insert(0, '/home/nij/GNU/main/lib/python2.7/dist-packages/pmt/')
sys.path.insert(0, '/home/nij/GNU/main/lib/')
parseHeader = imp.load_source('gr_read_file_metadata_nick', '/home/nij/GNU/main/bin/gr_read_file_metadata_nick')


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def crossCor(frame):
    ltf = numpy.array(
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
        dtype=numpy.complex64)
    corr = abs(numpy.correlate(frame, ltf, mode='same'))
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
    return startFrame


def test_plots(raw_frame, corr_frame):
    fig, ax = plt.subplots(4)
    fig.suptitle('CrossCorrelation, Magnitude, Power, Freq Plots')
    ax[0].plot(corr_frame, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x) ** 2, raw_frame)), 'r')
    ax[3].plot(abs(numpy.fft.fftshift(numpy.fft.fft(raw_frame))), 'g')
    plt.show()


def make_frame(args, raw_frame, mac_addr, frame_number):
    indiv_frame = []
    indiv_frame.append(mac_addr)
    if (args.debug):
        indiv_frame.append(frame_number)
    else:
        indiv_frame.append(str(uuid.uuid4()))
    indiv_frame.append(len(raw_frame))
    indiv_frame.append(raw_frame)
    if (args.got_csi):
        csi_data = open(args.base_path + "csi_data" + args.fileName + ".txt").read()
        linestart = csi_data.find(str(frame_number))
        csistart = csi_data.find("|", linestart, len(csi_data)) + 1
        csiend = csi_data.find("\n", csistart, len(csi_data))
        csi_frame = csi_data[csistart:csiend]
        indiv_frame.append(csi_frame)
        csi_data.close()
    return indiv_frame


def write2file(args, total_frames):
    # Sorts frames based on MAC Addresses
    # Writes frames to text file
    total_frames = list(filter(None, total_frames))
    if (not args.debug):
        total_frames.sort()

    finished_data = open(args.base_path + "finished_data" + args.fileName + ".txt", "w")

    if (args.got_csi):
        for each in total_frames:
            finished_data.write(
                str(each[0]) + "|" + str(each[1]) + "|" + str(each[2]) + "|" + str(each[3]).strip("[] ") + "|" + str(
                    each[4]).strip("[] ") + "\n")
    else:
        for each in total_frames:
            finished_data.write(
                str(each[0]) + "|" + str(each[1]) + "|" + str(each[2]) + "|" + str(each[3]).strip("[] ") + "\n")

    print("Done")
    # Difference should always be less than MAX FRAME SIZE in sync short(MAX FRAME SIZE with No Buffer at 20 MHz = 43200)
    print "Frames in File: ", len(total_frames)
    finished_data.close()


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
    raw_frame = []
    total_frames = []
    index = 0
    last_frame_number = 0
    start_num = 0
    curr_save = 0

    for header in header_data:

        # If not at end of parsed_data file, reads in next line
        # Frame_number is next frame recorded. If raw_frame is not sequentially in that place, it is skipped
        last_pos = parsed_data.tell()
        current_line = parsed_data.read()
        raw_frame_number = int(header[2])

        if (header[1] != 0):
            if (current_line != ""):
                parsed_data.seek(last_pos)
                mac_addr, frame_number = parsed_data.readline().split(",")
                frame_number = int(frame_number.strip("\n"))

            """Frame Number is the number associated with the frame in the parsed data text file. Raw Frame Number is the number
            saved in the metadata file associated with that raw frame. Both should always be increasing (although Frame Number can
            have duplicates). This if statement checks to see:
                1. if frame number is less than raw_frame_number (Should never happen) or is a duplicate, then skip that frame number and raw frame
                2. if frame number is greater than raw_frame_number (most common), then keep frame number and skip raw_frame
                3. Skip 1st 100 frames captured correctly and if any frames come from zeroed MAC addresses (potential virtual machines or spoofs)
                4. Proceed with saving successfully captured frame"""

            if (frame_number < raw_frame_number):
                raise Exception("Recording Error!!!!")
            elif (last_frame_number == frame_number):
                while (last_frame_number == frame_number):
                    mac_addr, frame_number = parsed_data.readline().split(",")
                    frame_number = int(frame_number.strip("\n"))
                parsed_data.seek(last_pos)
                index += header[1]
            elif (frame_number > raw_frame_number):
                parsed_data.seek(last_pos)
                index += header[1]
            elif (mac_addr == "00:0:0:0:0:0" or header[1] <= 10000):
                start_num += 1
                index += header[1]
            else:
                last_frame_number = frame_number
                # Checks to see if the target
                if (args.target_mac != mac_addr and args.target_mac):
                    index += header[1]
                    continue

                # Reads in raw frame and updates last frame number that has been read (used to remove duplicates above)
                for each_item in range(header[1]):
                    raw_frame.append(raw_data[index + each_item])
                # Calls cross correlation function and increments index for raw binary file
                # Also plots the test_plots to view the frame
                index += header[1]

                # Used to ensure there is not more than num_save frames stored
                if (curr_save < args.num_save):
                    curr_save += 1
                else:
                    break

                if (args.plot_frames and args.debug):
                    startFrame = crossCor(raw_frame)
                    test_plots(raw_frame, startFrame)
                elif (args.plot_frames and not args.debug):
                    test_plots(raw_frame, raw_frame)

                # If cross correlation function returned something (Success!) then save the MAC addr,
                # globally unique identifier, raw data length, raw data, and corresponding CSI data to array to be written to file
                total_frames.append(make_frame(args, raw_frame, mac_addr, frame_number))

        raw_frame = []

    # Checks to make sure no extra frames captured (Never been a problem before)
    parsed_data.close()
    if file_len(args.base_path + "parsed_data" + args.fileName + ".txt") < len(total_frames):
        raise Exception("Captured too many frames.")

    write2file(args, total_frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes 5 files (file sink, file sink header, parse data, and '
                                                 'CSI data and organizes them into a text file where each lline corresponds to a single frame',
                                     version=1.0)
    parser.add_argument('-lte', action='store_true', default=False, dest='parse_lte', help='Parses LTE files')
    parser.add_argument('-wifi', action='store_true', default=True, dest='parse_wifi', help='Parses WiFi')
    parser.add_argument('-file', action='store', default='', dest='fileName',
                        help='File prefix at the end of each file')
    parser.add_argument('-base_path', action='store', default="/home/nij/GNU/", dest='base_path',
                        help='Path to all files')
    parser.add_argument('-p', action='store_true', default=False, dest='plot_frames',
                        help='Plots frames in matplotlib to appear on screen')
    parser.add_argument('-n', action='store', default=float('inf'), dest='num_save',
                        help='The upper bound of how many frames will be saved to file')
    parser.add_argument('-d', action='store_true', default=False, dest='debug', help='Turns on debug mode')
    parser.add_argument('-no_csi', action='store_false', default=False, dest='got_csi', help='No CSI data for frames')
    parser.add_argument('-mac', action='store', default="", dest='target_mac',
                        help='MAC Address to specify the only device to be saved')
    args = parser.parse_args()

    if (args.parse_wifi):
        wifi_main(args)
    elif (args.parse_lte):
        lte_main(args)
    else:
        print "Choose to either parse wifi or lte"
