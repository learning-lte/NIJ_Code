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
#include <ieee802-11/sync_short.h>
#include <gnuradio/io_signature.h>
#include "utils.h"

#include <iostream>

using namespace gr::ieee802_11;

static const int MIN_GAP = 480;
static const int MAX_SAMPLES = 540 * 80;
int frame_counter = 0;
std::vector<gr_complex> d_raw_vec;

class sync_short_impl : public sync_short {

public:
sync_short_impl(double threshold, unsigned int min_plateau, unsigned int samp_fact, double start_buffer, double end_buffer, bool log, bool debug) :
		block("sync_short",
			gr::io_signature::makev(4, 4, get_input_sizes()),
			gr::io_signature::make(2, 2, sizeof(gr_complex))),
		d_log(log),
		d_debug(debug),
		d_state(SEARCH),
		d_plateau(0),
		d_freq_offset(0),
		d_copied(0),
		MIN_PLATEAU(min_plateau),
		d_threshold(threshold),
		d_samp_fact(samp_fact),
		d_start_buffer(start_buffer),
		d_end_buffer(end_buffer),
		d_frame_found(false){

	set_tag_propagation_policy(block::TPP_DONT);
}

int general_work (int noutput_items, gr_vector_int& ninput_items,
		gr_vector_const_void_star& input_items,
		gr_vector_void_star& output_items) {

	const gr_complex *in = (const gr_complex*)input_items[0];
	const gr_complex *in_abs = (const gr_complex*)input_items[1];
	const float *in_cor = (const float*)input_items[2];
	const gr_complex *in_delayed = (const gr_complex *)input_items[3];
	gr_complex *out = (gr_complex*)output_items[0];
	gr_complex *out1 = (gr_complex*)output_items[1];

	int noutput = noutput_items;
	int ninput = std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]);
	
	//dout << "Sampling Factor: " << d_samp_fact << std::endl;
	//dout << "SHORT noutput : " << noutput << " Orig ninput: " << ninput_items[0]  << " 40 MHZ ninput: " << ninput_items[3] << std::endl;

	switch(d_state) {

	case SEARCH: {
		int i;
		
		for(i = 0; i < ninput; i++) {
			if(in_cor[i] > d_threshold) {
				if(d_plateau < MIN_PLATEAU) {
					d_plateau++;

				} else {
					d_state = COPY;
					d_copied = 0;
					d_freq_offset = arg(in_abs[i]) / 16;
					d_plateau = 0;
					insert_tag(nitems_written(0), d_freq_offset, nitems_read(0) + i, 0);
					//dout << "SHORT Frame!" << std::endl;
					break;
				}
			} else {
				d_plateau = 0;
			}
		}

		consume_each(i);
		consume(3, i*(d_samp_fact -1));
		return 0;
	}

	case COPY: {

		int o = 0;
		
		while( o < ninput && o < noutput && d_copied < (MAX_SAMPLES + d_start_buffer + d_end_buffer)) {
			if(in_cor[o] > d_threshold) {
				if(d_plateau < MIN_PLATEAU) {
					d_plateau++;

				// there's another frame
				} else if(d_copied > MIN_GAP) {
					d_copied = 0;
					d_plateau = 0;
					d_freq_offset = arg(in_abs[o]) / 16;
					insert_tag(nitems_written(0) + o, d_freq_offset, nitems_read(0) + o, (d_samp_fact - 1) * o + 1);
					//dout << "Another SHORT Frame!" << std::endl;
					break;
				}

			} else {
				d_plateau = 0;
			}

			out[o] = in[o] * exp(gr_complex(0, -d_freq_offset * d_copied));
			for (int j = 0; j < d_samp_fact; j++){
				out1[o * d_samp_fact + j] = in_delayed[o * d_samp_fact + j];
				d_raw_vec.push_back(in_delayed[o * d_samp_fact + j]);
			}
			o++;
			d_copied++;
		}

		if(d_copied == MAX_SAMPLES + d_start_buffer + d_end_buffer) {
			d_state = SEARCH;
		}

		//dout << "SHORT copied " << o << " SHORT copied buffer " << (o1 + o*d_samp_fact) << std::endl;
		dout << "Number Read 20: " << nitems_read(0) << " Number Read 40: " << nitems_read(3) << std::endl;

		consume_each(o);
		consume(3, o*(d_samp_fact-1));
		
		produce(0, o);
		produce(1, o*d_samp_fact);
		o = 0;
		return WORK_CALLED_PRODUCE;
	}
	}

	throw std::runtime_error("sync short: unknown state");
	return 0;
}


void insert_tag(uint64_t item, double freq_offset, uint64_t input_item, int o) {
	mylog(boost::format("frame start at in: %2% out: %1% out40: %3%") % item % input_item % nitems_read(3));
	frame_counter += 1;
	gr_complex raw_arr[d_raw_vec.size()];
	std::copy(d_raw_vec.begin(), d_raw_vec.end(), raw_arr);
	uintptr_t temp = reinterpret_cast<uintptr_t>(raw_arr);
	uint64_t raw_size = static_cast<uint64_t>(d_raw_vec.size());
	const pmt::pmt_t key = pmt::string_to_symbol("wifi_start");
	const pmt::pmt_t value = pmt::from_double(freq_offset);
	const pmt::pmt_t value1 = pmt::from_uint64(frame_counter);
	pmt::pmt_t srcid = pmt::make_dict();
	srcid = pmt::dict_add(srcid, pmt::mp("framecounter"), pmt::string_to_symbol(std::to_string(frame_counter)));
	srcid = pmt::dict_add(srcid, pmt::mp("raw_data"), pmt::from_uint64(temp));
	srcid = pmt::dict_add(srcid, pmt::mp("raw_size"), pmt::from_uint64(raw_size));
	add_item_tag(0, item, key, value, srcid);
	add_item_tag(1, (item - nitems_written(0) + nitems_written(1) + o * (d_samp_fact - 1)), key, value1, srcid);
	d_raw_vec.clear();
	std::cout << "SS: Raw Data: " << raw_arr[0]  << "," << frame_counter << "," << raw_arr << std::endl;
}


static std::vector<int> get_input_sizes(){
     return {
         sizeof(gr_complex),
         sizeof(gr_complex),
         sizeof(float),
         sizeof(gr_complex),
     };
}

private:
	enum {SEARCH, COPY} d_state;
	int d_copied;
	int d_plateau;
	float d_freq_offset;
	const double d_threshold;
	const bool d_log;
	const bool d_debug;
	const unsigned int MIN_PLATEAU;
	const  double d_start_buffer;
	const double d_end_buffer;
	bool d_frame_found;
	const unsigned int d_samp_fact;
};

sync_short::sptr
sync_short::make(double threshold, unsigned int min_plateau, unsigned int samp_fact, double start_buffer, double end_buffer, bool log, bool debug) {
	return gnuradio::get_initial_sptr(new sync_short_impl(threshold, min_plateau, samp_fact, start_buffer, end_buffer, log, debug));
}