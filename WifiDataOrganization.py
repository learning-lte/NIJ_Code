import imp
import scipy
import numpy
import sys
import uuid
import matplotlib.pyplot as plt
#from gnuradio.blocks import parse_file_metadata


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def autoCorMatch(frame):
    autoCorList = []

    if (len(frame) > 15000):

        # Calculate first sample's autoCorrelation and alter to get rest
        current_auto = 0
        current_power = 0
        for index2 in range(48 + 15):
            current_auto += frame[index2] * numpy.conj(frame[index2 + 16])
        for index3 in range(48 - 1):
            current_power += frame[index3] * numpy.conj(frame[index3])

        autoSample = abs(current_auto) / current_power
        autoAverage = autoSample
        autoMax = autoSample

        # Alter with autoCorrelate function
        for sampIndex in range(len(frame)):
            if (sampIndex == len(frame) - 64 - 16):
                break
            current_auto -= (frame[sampIndex] * numpy.conj(frame[sampIndex + 16]))
            current_power -= frame[sampIndex] * numpy.conj(frame[sampIndex])
            current_auto += (frame[sampIndex + 64] * numpy.conj(frame[sampIndex + 64 + 16]))
            current_power += frame[sampIndex + 48] * numpy.conj(frame[sampIndex + 48])
            autoSample = (abs(current_auto) / current_power)
            autoCorList.append(autoSample)
            if (autoSample > autoMax):
                autoMax = autoSample
            if (autoSample < 0):
                return None
            autoAverage += autoSample
        autoAverage = autoAverage / len(autoCorList)
        if autoMax - autoAverage > 1 and autoMax > 50:
            return autoCorList

    return None


def main():
    fileName = ""
    sys.path.insert(0, '/home/nij/GNU/main/lib/python2.7/dist-packages/')
    sys.path.insert(0, '/home/nij/GNU/main/lib/python2.7/dist-packages/pmt/')
    sys.path.insert(0, '/home/nij/GNU/main/lib/')
    parseHeader = imp.load_source('gr_read_file_metadata_nick', '/home/nij/GNU/main/bin/gr_read_file_metadata_nick')

    base_path = "/home/nij/GNU/"
    raw_data = scipy.fromfile(open(base_path + "file_sink" + fileName), dtype=scipy.complex64)
    parsed_data = open(base_path + "parsed_data" + fileName + ".txt", "r")
    csi_data = open(base_path + "csi_data" + fileName + ".txt").read()
    finished_data = open(base_path + "finished_data" + fileName + ".txt", "w")
    raw_frame = []
    indiv_frame = []
    total_frames = []
    # Returns a 2D array containing raw data header info sorted by frame
    # Filters combined data for NULL frames
    # All units are bytes
    # Format: HEADER LENGTH, SIZE IN Items, Frame Counter

    header_data = parseHeader.main(base_path + "file_sink" + fileName + ".hdr", True)
    header_data = list(filter(None, header_data))

    # Sort raw data into frames based on header info
    # Combines parsed data, raw data and assigns frames ID if the raw frame wasnt skipped
    # Parsed Data Format: Transmitter MAC, Frame Number (Counter)
    # Final Format: MAC ADDRESS, FRAME ID, FRAME LENGTH (in items), [RAW DATA]
    index = 0
    last_raw_counter = 0
    last_frame_number = 0
    start_num = 0
    freq_list = []
    num_save = 3000
    curr_save = 0

    for header in header_data:

        # If not at end of parsed_data file, reads in next line
        # Frame_number is next frame recorded. If raw_frame is not sequentially in that place, it is skipped
        last_pos = parsed_data.tell()
        current_line = parsed_data.read()
        raw_frame_counter = int(header[2])

        if (header[1] != 0):
            if (current_line != ""):
                parsed_data.seek(last_pos)
                mac_addr, frame_number = parsed_data.readline().split(",")
                frame_number = int(frame_number.strip("\n"))

                # This while loop gets rid of duplicate frame numbers in parsed data by
                # repeatedly reading lines until the next frame number is not equal to the last saved
                while (last_frame_number == frame_number):
                    mac_addr, frame_number = parsed_data.readline().split(",")
                    frame_number = int(frame_number.strip("\n"))
                    last_pos = parsed_data.tell()

            """Frame Number is the number associated with the frame in the parsed data text file. Raw Frame counter is the number
            saved in the metadata file associated with that raw frame. Both should always be increasing (although Frame Number can
            have duplicates). This if statement checks to see:
                1. Data was saved correctly in the raw binary file
                2. if frame number is less than raw_frame_counter (Should never happen), then skip the raw_frame and that frame number
                3. if frame number is greater than raw_frame_counter (most common), then keep frame number and skip raw_frame
                4. Skip 1st 100 frames captured correctly and if any frames come from zeroed MAC addresses (potential virtual machines or spoofs)
                5. Proceed with saving successfully captured frame"""

            if (raw_frame_counter != (last_raw_counter + 1)):
                raise Exception("Error in raw data of collected frames at ", raw_frame_counter)
                pass
            elif (frame_number < raw_frame_counter):
                index += header[1]
            elif (frame_number > raw_frame_counter):
                parsed_data.seek(last_pos)
                index += header[1]
            elif (start_num <= 10 or mac_addr == "00:0:0:0:0:0" or header[1] < 10000):
                start_num += 1
                index += header[1]
            else:
                # Reads in raw frame and updates last frame number that has been read (used to remove duplicates above)
                last_frame_number = frame_number
                for each_item in range(header[1]):
                    raw_frame.append(raw_data[index + each_item])
                # Calls autocorrelation function and increments index for raw binary file
                #test1 = autoCorMatch(raw_frame)
                index += header[1]
                # Used to ensure there is not more than 100 frames captured from each device
                flag1 = False
                flag2 = False
                #if (test1 is not None):
                if (mac_addr == "f4:71:90:a1:d1:2c"):
                    if (curr_save < num_save):
                        flag2 = True
                        curr_save += 1
                    else:

                        break
                    """for addr in freq_list:
                        if mac_addr == addr[0]:
                            if addr[1] <= num_save:
                                addr[1] += 1
                                flag2 = True
                            flag1 = True
                    if not flag1:
                        freq_list.append([mac_addr, 1])"""

                # If autocorrelation function returned something (Success!) then save the MAC addr,
                # globally unique identifier, raw data length, raw data, and corresponding CSI data to array to be written to file
                ####test1 is not None and
                #if (test1 is not None and flag2):
                if (flag2):
                    indiv_frame.append(mac_addr)
                    #indiv_frame.append(frame_number)
                    indiv_frame.append(str(uuid.uuid4()))
                    indiv_frame.append(header[1])
                    indiv_frame.append(raw_frame)
                    linestart = csi_data.find(str(frame_number))
                    csistart = csi_data.find("|", linestart, len(csi_data)) + 1
                    csiend = csi_data.find("\n", csistart, len(csi_data))
                    csi_frame = csi_data[csistart:csiend]
                    indiv_frame.append(csi_frame)
                    total_frames.append(indiv_frame)
                    ###Used for testing
                    #test_plots(raw_frame, test1)

        last_raw_counter = raw_frame_counter
        raw_frame = []
        indiv_frame = []

    parsed_data.close()
    if file_len(base_path + "parsed_data" + fileName + ".txt") < len(total_frames):
        raise Exception("Captured too many frames.")

    # Sorts frames based on MAC Addresses
    # Writes frames to text file
    # total_frames = list(filter(None, total_frames))
    # total_frames.sort()

    for each in total_frames:
        finished_data.write(
            str(each[0]) + "|" + str(each[1]) + "|" + str(each[2]) + "|" + str(each[3]).strip("[] ") + "|" + str(each[4]).strip("[] ") + "\n")

    print("Done")
    # Difference should always be less than MAX FRAME SIZE (No Buffer at 20 MHz = 43200)
    print("Frames in File: ", len(total_frames))
    print(freq_list)

    finished_data.close()


def test_plots(raw_frame, test1):
    fig, ax = plt.subplots(4)
    fig.suptitle('Autocorrelation, Magnitude, Power, Freq Plots')
    ax[0].plot(test1, 'y')
    ax[1].plot(list(map(lambda x: abs(x), raw_frame)), 'b')
    ax[2].plot(list(map(lambda x: abs(x) ** 2, raw_frame)), 'r')
    ax[3].plot(abs(numpy.fft.fftshift(numpy.fft.fft(raw_frame))), 'g')
    plt.show()


main()
