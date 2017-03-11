#include "neuron.h"
#include <iostream>
#include <iomanip>

neuron::neuron()
{
	rate=0.01;
	num=3;
	weights = new double[3];
	for(int i=0;i<num;i++)
		weights[i]=static_cast<double>(rand())/RAND_MAX*2-1.0;
}

neuron::neuron(int number, double r)
{
	std::cout << "Neuron created -- Leanring rate: " << r << "   Number of input: " << number << std::endl;
	rate=r;
	num=number;
	weights = new double[number];
	for(int i=0;i<number;i++)
	{
		weights[i]=static_cast<double>(rand())/RAND_MAX*2-1.0;
		//std::cout << std::setw(10) << weights[i] << "   ";
	}
	//std::cout << "\n" << std::endl;
}

neuron::~neuron()
{
	delete[] weights;
	std::cout << "Neuron destroyed" << std::endl;
}

double neuron::feedforward(double inputs[]) const
{
	double sum=0.0;
	for(int i=0;i<num;i++)
		sum+=inputs[i]*weights[i];
	return 1/(1+exp(-sum));
	//return sum/(2*(1+(sum>0?sum:-sum))) + 0.5;
}

void neuron::update_weight(double independent, double dependent[])
{
	for(int i=0;i<num-1;i++)
		weights[i] += rate * independent * dependent[i];
	weights[num-1] += rate * independent * 1;
}

void neuron::get_weight(double array[]) const
{
	for(int i=0;i<num;i++)
		array[i]=weights[i];
}

