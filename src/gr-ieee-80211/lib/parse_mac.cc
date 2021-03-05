/*
 * Copyright (C) 2013, 2016 Bastian Bloessl <bloessl@ccs-labs.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <ieee802-11/parse_mac.h>
#include "utils.h"

#include <gnuradio/io_signature.h>
#include <gnuradio/block_detail.h>
#include <string>
using namespace gr::ieee802_11;
uint64_t frames_captured = 0;

class parse_mac_impl : public parse_mac {

public:

parse_mac_impl(bool log, bool debug, const std::string& mac, const std::string& filename) :
		block("parse_mac",
				gr::io_signature::make(0, 0, 0),
				gr::io_signature::make(0, 0, 0)),
		d_log(log), d_last_seq_no(-1),
		d_debug(debug), d_mac(mac), d_filename(filename) {

	message_port_register_in(pmt::mp("in"));
	set_msg_handler(pmt::mp("in"), boost::bind(&parse_mac_impl::parse, this, _1));

	message_port_register_out(pmt::mp("fer"));
}

~parse_mac_impl() {

}

void parse(pmt::pmt_t msg) {
	if(pmt::is_eof_object(msg)) {
		detail().get()->set_done(true);
		return;
	} else if(pmt::is_symbol(msg)) {
		return;
	}
		
	pmt::pmt_t dict = pmt::car(msg);
	pmt::pmt_t temp = pmt::dict_ref(dict, pmt::mp("framecounter"), pmt::string_to_symbol("0"));
	std::string d_frame_counter = pmt::symbol_to_string(temp);

	msg = pmt::cdr(msg);
	int data_len = pmt::blob_length(msg);
	mac_header *h = (mac_header*)pmt::blob_data(msg);	

	mylog(boost::format("length: %1%") % data_len );

	dout << std::endl << "new mac frame  (length " << data_len << ")" << std::endl;
	dout << "=========================================" << std::endl;
	if(data_len < 20) {
		dout << "frame too short to parse (<20)" << std::endl;
		return;
	}
	#define HEX(a) std::hex << std::setfill('0') << std::setw(2) << int(a) << std::dec
	dout << "duration: " << HEX(h->duration >> 8) << " " << HEX(h->duration  & 0xff) << std::endl;
	dout << "frame control: " << HEX(h->frame_control >> 8) << " " << HEX(h->frame_control & 0xff);

		switch((h->frame_control >> 2) & 3) {

		case 0:
			dout << " (MANAGEMENT)" << std::endl;
			parse_management((char*)h, data_len, d_frame_counter);
			break;
		case 1:
			dout << " (CONTROL)" << std::endl;
			parse_control((char*)h, data_len, d_frame_counter);
			break;

		case 2:
			dout << " (DATA)" << std::endl;
			parse_data((char*)h, data_len, d_frame_counter);
			break;

		default:
			dout << " (unknown)" << std::endl;
			break;
	}

	char *frame = (char*)pmt::blob_data(msg);

	// DATA
	if((((h->frame_control) >> 2) & 63) == 2) {
		print_ascii(frame + 24, data_len - 24);
	// QoS Data
	} else if((((h->frame_control) >> 2) & 63) == 34) {
		print_ascii(frame + 26, data_len - 26);
	}
}

void parse_management(char *buf, int length, std::string d_frame_counter) {

	mac_header* h = (mac_header*)buf;

	if(length < 24) {
		dout << "too short for a management frame" << std::endl;
		return;
	}

	dout << "Subtype: ";
	switch(((h->frame_control) >> 4) & 0xf) {
		case 0:
			dout << "Association Request";
			break;
		case 1:
			dout << "Association Response";
			break;
		case 2:
			dout << "Reassociation Request";
			break;
		case 3:
			dout << "Reassociation Response";
			break;
		case 4:
			dout << "Probe Request";
			break;
		case 5:
			dout << "Probe Response";
			break;
		case 6:
			dout << "Timing Advertisement";
			break;
		case 7:
			dout << "Reserved";
			break;
		case 8:
			dout << "Beacon" << std::endl;
			if(length < 38) {
				return;
			}
			{
			uint8_t* len = (uint8_t*) (buf + 24 + 13);
			if(length < 38 + *len) {
				return;
			}
			std::string s(buf + 24 + 14, *len);
			dout << "SSID: " << s;
			}
			break;
		case 9:
			dout << "ATIM";
			break;
		case 10:
			dout << "Disassociation";
			break;
		case 11:
			dout << "Authentication";
			break;
		case 12:
			dout << "Deauthentication";
			break;
		case 13:
			dout << "Action";
			break;
		case 14:
			dout << "Action No ACK";
			break;
		case 15:
			dout << "Reserved";
			break;
		default:
			break;
	}
	dout << std::endl;

	dout << "seq nr: " << int(h->seq_nr >> 4) << std::endl;
	dout << "mac 1: ";
	////////////////////////////////////////////////////////////////////////////////////////
	print_mac_address(h->addr1, true, d_frame_counter, false);
	dout << "mac 2: ";
	print_mac_address(h->addr2, true, d_frame_counter, true);
	dout << "mac 3: ";
	print_mac_address(h->addr3, true, d_frame_counter, false);
	dout << "Number of Frames Captured: " << frames_captured << std::endl;
	///////////////////////////////////////////////////////////////////////////////////////

}


void parse_data(char *buf, int length, std::string d_frame_counter) {

	mac_header* h = (mac_header*)buf;
	if(length < 24) {
		dout << "too short for a data frame" << std::endl;
		return;
	}

	dout << "Subtype: ";
	switch(((h->frame_control) >> 4) & 0xf) {
		case 0:
			dout << "Data";
			break;
		case 1:
			dout << "Data + CF-ACK";
			break;
		case 2:
			dout << "Data + CR-Poll";
			break;
		case 3:
			dout << "Data + CF-ACK + CF-Poll";
			break;
		case 4:
			dout << "Null";
			break;
		case 5:
			dout << "CF-ACK";
			break;
		case 6:
			dout << "CF-Poll";
			break;
		case 7:
			dout << "CF-ACK + CF-Poll";
			break;
		case 8:
			dout << "QoS Data";
			break;
		case 9:
			dout << "QoS Data + CF-ACK";
			break;
		case 10:
			dout << "QoS Data + CF-Poll";
			break;
		case 11:
			dout << "QoS Data + CF-ACK + CF-Poll";
			break;
		case 12:
			dout << "QoS Null";
			break;
		case 13:
			dout << "Reserved";
			break;
		case 14:
			dout << "QoS CF-Poll";
			break;
		case 15:
			dout << "QoS CF-ACK + CF-Poll";
			break;
		default:
			break;
	}
	dout << std::endl;

	int seq_no = int(h->seq_nr >> 4);
	dout << "seq nr: " << seq_no << std::endl;
	dout << "mac 1: ";
	////////////////////////////////////////////////
	print_mac_address(h->addr1, true, d_frame_counter, false);
	dout << "mac 2: ";
	print_mac_address(h->addr2, true, d_frame_counter, true);
	dout << "mac 3: ";
	print_mac_address(h->addr3, true, d_frame_counter, false);
	dout << "Number of Frames Captured: " << frames_captured << std::endl;
	/////////////////////////////////////////////////

	float lost_frames = seq_no - d_last_seq_no - 1;
	if(lost_frames  < 0)
		lost_frames += 1 << 12;

	// calculate frame error rate
	float fer = lost_frames / (lost_frames + 1);
	dout << "instantaneous fer: " << fer << std::endl;

	// keep track of values
	d_last_seq_no = seq_no;

	// publish FER estimate
	pmt::pmt_t pdu = pmt::make_f32vector(lost_frames + 1, fer * 100);
	message_port_pub(pmt::mp("fer"), pmt::cons( pmt::PMT_NIL, pdu ));
}

void parse_control(char *buf, int length, std::string d_frame_counter) {

	mac_header* h = (mac_header*)buf;

	dout << "Subtype: ";
	switch(((h->frame_control) >> 4) & 0xf) {
		case 7:
			dout << "Control Wrapper";
			break;
		case 8:
			dout << "Block ACK Requrest";
			break;
		case 9:
			dout << "Block ACK";
			break;
		case 10:
			dout << "PS Poll";
			break;
		case 11:
			dout << "RTS";
			break;
		case 12:
			dout << "CTS";
			break;
		case 13:
			dout << "ACK";
			break;
		case 14:
			dout << "CF-End";
			break;
		case 15:
			dout << "CF-End + CF-ACK";
			break;
		default:
			dout << "Reserved";
			break;
	}
	dout << std::endl;

	dout << "RA: ";
	/////////////////////////////
	print_mac_address(h->addr1, true, d_frame_counter, false);
	dout << "TA: ";
	print_mac_address(h->addr2, true, d_frame_counter, true);
	dout << "Number of Frames Captured: " << frames_captured << std::endl;
	/////////////////////////////

}

void print_mac_address(uint8_t *addr, bool new_line = false, std::string d_frame_counter = "", bool target = false) {

	if(!d_debug) {
		return;
	}

	std::cout << std::setfill('0') << std::hex << std::setw(2);
	
	if (target == true and last_frame_counter != curr_frame_counter and last_raw_addr != curr_raw_addr and curr_raw_ant2 != last_raw_ant2 and curr_raw_addr->size() > 0){

		std::stringstream mac_addr;
		mac_addr << std::setfill('0') << std::hex << std::setw(2);
	
		for(int i = 0; i < 6; i++) {
		mac_addr << (int)addr[i];
			if(i != 5) {
				mac_addr << ":";
			}
		}
	
		if (!d_mac.empty()){
			if (mac_addr.str()  == d_mac){
				save_parsed(d_frame_counter, mac_addr.str());
			}
		}
		else{
			save_parsed(d_frame_counter, mac_addr.str());
		}
	}

	for(int i = 0; i < 6; i++) {
		std::cout << (int)addr[i];
		if(i != 5) {
			std::cout << ":";
		}
	}

	std::cout << std::dec;
	
	if(new_line) {
		std::cout << std::endl;
	}
}


	/*List of Test Mac Addresses
	"00:8:22:34:bc:fb"   <======Changes on Reboot. Check in phone
	"f4:71:90:a1:d1:2c" 
	"96:e2:bd:d4:ba:87"<=======
	"68:c4:4d:97:89:9e"
	"e8:3e:b6:3d:1c:c9"
	"d0:13:fd:63:63:87"
	"e0:5f:45:73:3d:f1"
	"f0:79:60:7d:a2:12"
	"a0:d7:95:1f:75:f7"
	*/ 
void save_parsed(std::string d_frame_counter, std::string mac_addr){
	std::ofstream parsed_data;
	parsed_data.open(d_filename, std::ios::out | std::ios::app);
	parsed_data << std::setfill('0') << std::hex << std::setw(2);
	parsed_data << mac_addr;
	parsed_data << std::dec;
	parsed_data << "," << d_frame_counter << "\n";
	parsed_data.close();
	frames_captured ++;
}

void print_ascii(char* buf, int length) {

	for(int i = 0; i < length; i++) {
		if((buf[i] > 31) && (buf[i] < 127)) {
			dout << buf[i];
		} else {
			dout << ".";
		}
	}
	dout << std::endl;
}

private:
	bool d_log;
	bool d_debug;
	const std::string d_mac;
	const std::string d_filename;
	int d_last_seq_no;
};

parse_mac::sptr
parse_mac::make(bool log, bool debug, const std::string& mac, const std::string& filename) {
	return gnuradio::get_initial_sptr(new parse_mac_impl(log, debug, mac, filename));
}


