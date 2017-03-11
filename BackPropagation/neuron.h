#ifndef _NEURON_H_
#define _NEURON_H_

#include <cstdlib>
#include <ctime>
#include <cmath>

class neuron
{
	public:
		neuron();
		neuron(int,double);
		~neuron();
		void update_weight(double, double []);
		
		double feedforward(double []) const;
		void get_weight(double []) const;
		
	private:
		double rate;
		int num;
		double* weights;
		
};

#endif
