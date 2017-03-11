#ifndef LAYER_H_
#define LAYER_H_

#include <vector>
#include "neuron.h"

class Layer
{
	public:
		Layer();
		Layer(int, int, double);
		~Layer();
		
		void feedforward(double[]);
		void update_weights(double[], double[]);
		
		void get_outputs(double[]) const;
		
		void get_weights(int, double[]) const;
		
	private:
		
		std::vector<neuron*> neuronVector;
		
		int num;
		int neuron_num;
		double rate;
		double* outputs;
};

#endif
