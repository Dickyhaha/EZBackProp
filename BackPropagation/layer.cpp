#include "layer.h"
#include <iostream>

Layer::Layer()
{
	num=3;
	neuron_num=3;
	rate=0.01;
	outputs = new double[3];
	for(int i=0;i<3;i++)
	{
		outputs[i]=0.0;
		neuronVector.push_back(new neuron(num,rate));
	}
}

Layer::Layer(int number,int neuron_no, double rates)
{
	std::cout << "\nLayer created: " << number << " " << neuron_no << std::endl;
	num=number+1;	//Adding bias node
	neuron_num=neuron_no;
	rate=rates;
	outputs = new double[neuron_no];
	for(int i=0;i<neuron_no;i++)
	{
		outputs[i]=0.0;
		neuronVector.push_back(new neuron(number+1,rate));
	}
}

Layer::~Layer()
{
	delete[] outputs;
	outputs=0;
	for(int i=0;i<neuron_num;i++)
		delete neuronVector[i];
	std::cout << "\nLayer destroyed\n" << std::endl;
}

void Layer::feedforward(double inputs[])
{
	for(int i=0;i<neuron_num;i++)
		outputs[i]=neuronVector[i]->feedforward(inputs);
}

void Layer::update_weights(double independents[], double dependents[])
{
	for(int i=0;i<neuron_num;i++)
		neuronVector[i]->update_weight(independents[i],dependents);
}

void Layer::get_outputs(double array[]) const
{
	for(int i=0;i<neuron_num;i++)
		array[i]=outputs[i];
}

void Layer::get_weights(int i, double array[]) const
{
	neuronVector[i]->get_weight(array);
}






