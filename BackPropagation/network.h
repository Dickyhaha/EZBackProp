#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

class Network
{
	public:
		Network();
		Network(const int, const int, const int[], const int, const double);
		~Network();
		
		void feedforward(double[]);
		void feedback(double[]);
		
		void print_weights() const;
        void save_weights() const;
		void get_outputs(double[]) const;
		void get_output_errors(double[]) const;
		void get_hidden_outputs(int, double[]) const;
		void get_hidden_errors(int, double[]) const;
		
	private:
		int input_num;
		int hidden_layer_num;
		int* hidden_num;
		int output_num;
		double learning_rate;
		
		Layer** hidden;
		Layer* output;
		
		double* inputs;
		double** hidden_outputs;
		double** hidden_output_temp;
		double* outputs;
		
		double* output_errors;
		double** hidden_errors;
		
		double** prev_output_weights;
		double*** prev_hidden_weights;	
		
};

#endif
