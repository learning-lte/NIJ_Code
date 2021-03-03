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
long last_frame_counter = 0;
std::vector<gr_complex>* last_raw_addr = NULL;
std::vector<gr_complex>* last_raw_ant2 = NULL;
std::vector<uint64_t> old_addresses;
//std::ofstream parsed_data("/home/nick/GNU/parsed_data", std::ios::out | std::ios::trunc | std::ios::binary);

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
	pmt::pmt_t nij_dict = pmt::dict_ref(dict, pmt::mp("nij_dict"), pmt::PMT_NIL);
	if (pmt::is_null(nij_dict)){
		std::cout << "WTF" << std::endl;
	}
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
			parse_management((char*)h, data_len, nij_dict);
			break;
		case 1:
			dout << " (CONTROL)" << std::endl;
			parse_control((char*)h, data_len, nij_dict);
			break;

		case 2:
			dout << " (DATA)" << std::endl;
			parse_data((char*)h, data_len, nij_dict);
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

void parse_management(char *buf, int length, pmt::pmt_t nij_dict) {

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

	int seq_no = int(h->seq_nr >> 4);
	dout << "seq nr: " << seq_no << std::endl;
	dout << "mac 1: ";
	////////////////////////////////////////////////////////////////////////////////////////
	print_mac_address(h->addr1, true, nij_dict, false, seq_no);
	dout << "mac 2: ";
	print_mac_address(h->addr2, true, nij_dict, true, seq_no);
	dout << "mac 3: ";
	print_mac_address(h->addr3, true, nij_dict, false, seq_no);
	std::cout << std::dec << "Number of Frames Captured: " << frames_captured << std::endl;
	///////////////////////////////////////////////////////////////////////////////////////

}


void parse_data(char *buf, int length, pmt::pmt_t nij_dict) {

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
	print_mac_address(h->addr1, true, nij_dict, false, seq_no);
	dout << "mac 2: ";
	print_mac_address(h->addr2, true, nij_dict, true, seq_no);
	dout << "mac 3: ";
	print_mac_address(h->addr3, true, nij_dict, false, seq_no);
	std::cout << std::dec << "Number of Frames Captured: " << frames_captured << std::endl;
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

void parse_control(char *buf, int length, pmt::pmt_t nij_dict) {

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
	print_mac_address(h->addr1, true, nij_dict, false);
	dout << "TA: ";
	print_mac_address(h->addr2, true, nij_dict, false);
	/////////////////////////////

}

void print_mac_address(uint8_t *addr, bool new_line = false, pmt::pmt_t nij_dict = pmt::PMT_NIL, bool target = false, int seq_num=-1) {

/* 	if(!d_debug) {
		return;
	} */

	std::cout << std::setfill('0') << std::hex << std::setw(2);
	long curr_frame_counter = pmt::to_long(pmt::dict_ref(nij_dict, pmt::mp("framecounter"), pmt::from_long(0)));
	uint64_t curr_temp = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("raw_data"), pmt::from_uint64(0)));
	uint64_t curr_ant2 = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("raw_ant2"), pmt::from_uint64(0)));
	std::vector<gr_complex>* curr_raw_addr = reinterpret_cast<std::vector<gr_complex>*> (curr_temp);
	std::vector<gr_complex>* curr_raw_ant2 = reinterpret_cast<std::vector<gr_complex>*> (curr_ant2);
	
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
				save_parsed(nij_dict, mac_addr.str(), seq_num);
				last_frame_counter = curr_frame_counter;
				last_raw_addr = curr_raw_addr;
				last_raw_ant2 = curr_raw_ant2;
			}
		}
		else{
			save_parsed(nij_dict, mac_addr.str(), seq_num);
			last_frame_counter = curr_frame_counter;
			last_raw_addr = curr_raw_addr;
			last_raw_ant2 = curr_raw_ant2;
		}
		std::cout << std::dec;
	}

/* 	for(int i = 0; i < 6; i++) {
		std::cout << (int)addr[i];
		if(i != 5) {
			std::cout << ":";
		}
	}

	std::cout << std::dec;
	
	if(new_line) {
		std::cout << std::endl;
	} */
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
void save_parsed(pmt::pmt_t nij_dict, std::string mac_addr, int seq_num){
	std::ofstream parsed_data(d_filename, std::ios::out | std::ios::app | std::ios::binary);
	int d_frame_counter = pmt::to_long(pmt::dict_ref(nij_dict, pmt::mp("framecounter"), pmt::string_to_symbol("0")));
	//uint64_t raw_size = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("raw_size"), pmt::from_uint64(0)));
	uint64_t raw_temp = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("raw_data"), pmt::from_uint64(0)));
	uint64_t raw_ant2 = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("raw_ant2"), pmt::from_uint64(0)));
	uint64_t time_ns = pmt::to_uint64(pmt::dict_ref(nij_dict, pmt::mp("time_ns"), pmt::from_uint64(0)));
	std::vector<gr_complex>* raw_addr = reinterpret_cast<std::vector<gr_complex>*> (raw_temp);
	std::vector<gr_complex>* raw_addr_ant2 = reinterpret_cast<std::vector<gr_complex>*> (raw_ant2);
	int raw_size = raw_addr->size();
	int raw_ant2_size = raw_addr_ant2->size();
	char delim = '|';
/* 	const char* mac = mac_addr.c_str();
	parsed_data.write(mac, mac_addr.size());
	parsed_data.write(&delim, sizeof(char)); */
	parsed_data.write(reinterpret_cast<const char *> (&d_frame_counter), sizeof(int));
	parsed_data.write(&delim, sizeof(char));
	parsed_data.write(reinterpret_cast<const char *> (&seq_num), sizeof(int));
	parsed_data.write(&delim, sizeof(char));
	parsed_data.write(reinterpret_cast<const char *> (&time_ns), sizeof(uint64_t));
	parsed_data.write(&delim, sizeof(char));
	parsed_data.write(reinterpret_cast<const char *> (&raw_size), sizeof(int));
	parsed_data.write(&delim, sizeof(char));
	parsed_data.write(reinterpret_cast<const char *> (&raw_ant2_size), sizeof(int));
	parsed_data.write(&delim, sizeof(char));
	//parsed_data << '|' << d_frame_counter << '|' << seq_num << '|' << time_ns << '|';
	//std::vector<gr_complex>::iterator ptr;
/*  	for (ptr = raw_addr->begin(); ptr < raw_addr->end(); ptr++){
		parsed_data <<  std::showpos << std::setprecision(6) << '(' <<  std::real(*ptr) << std::imag(*ptr) << "j), ";
	} */
	//std::cout << "Ant 1: " << raw_addr->front() << " Ant 2: " << raw_addr_ant2->front() << '\n';
	parsed_data.write(reinterpret_cast<const char *> (raw_addr->data()), raw_size * sizeof(gr_complex));
	parsed_data.write(&delim, sizeof(char));
	parsed_data.write(reinterpret_cast<const char *> (raw_addr_ant2->data()), raw_ant2_size * sizeof(gr_complex));
	parsed_data << '\n';
	frames_captured ++;
	parsed_data.close();
	//dout << '\n' << "PM: Raw Data: " << raw_addr->size()  << ", " << d_frame_counter << "," << raw_addr << '\n';
	if (std::find(old_addresses.begin(), old_addresses.end(), raw_temp) != old_addresses.end() and std::find(old_addresses.begin(), old_addresses.end(), raw_ant2) != old_addresses.end()){
	delete raw_addr;
	delete raw_addr_ant2;
	old_addresses.push_back(raw_temp);
	old_addresses.push_back(raw_ant2);
	}
}

void print_ascii(char* buf, int length) {

	for(int i = 0; i < length; i++) {
		if((buf[i] > 31) && (buf[i] < 127)) {
			dout << buf[i];
		} else {
			dout << ".";
		}
	}
	dout << "\n";
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


