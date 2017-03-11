#include "network.h"
#include <iostream>
#include <iomanip>
#include <fstream>
using namespace std;

Network::Network()
{
    cout << "Network created";

    input_num = 2;
    hidden_layer_num=1;
    hidden_num = new int[hidden_layer_num];
    hidden_num[0] = 3;
    output_num = 2;
    learning_rate = 0.01;

    hidden = new Layer*[hidden_layer_num];
    hidden[0] = new Layer(2,3,0.01);
    output = new Layer(3,2,0.01);

    inputs = new double[input_num+1];
    hidden_outputs = new double*[hidden_layer_num];
    hidden_output_temp = new double*[hidden_layer_num];
    for(int i=0;i<hidden_layer_num;i++)
    {
        hidden_outputs[i] = new double[hidden_num[i]+1];
        hidden_output_temp[i] = new double[hidden_num[i]];
    }
    outputs = new double[output_num];

    inputs[input_num]=1;
    for(int i=0;i<hidden_layer_num;i++)
        hidden_outputs[i][hidden_num[i]]=1;

    hidden_errors = new double*[hidden_layer_num];
    for(int i=0;i<hidden_layer_num;i++)
        hidden_errors[i] = new double[hidden_num[i]];
    output_errors = new double[output_num];

    prev_hidden_weights = new double**[hidden_layer_num];
    prev_hidden_weights[0] = new double*[hidden_num[0]];
    for(int j=0;j<hidden_num[0];j++)
        prev_hidden_weights[0][j] = new double[input_num+1];
    for(int i=1;i<hidden_layer_num;i++)
    {
        prev_hidden_weights[i] = new double*[hidden_num[i]];
        for(int j=0;j<hidden_num[i];j++)
            prev_hidden_weights[i][j] = new double[hidden_num[i-1]+1];
    }

    prev_output_weights = new double*[output_num];
    for(int i=0;i<output_num;i++)
        prev_output_weights[i] = new double[hidden_num[hidden_layer_num-1]+1];
}

Network::Network(const int num_input, const int num_hidden_layer, const int num_hidden[], const int num_output, const double rate)
{

    cout << "Network created" << endl;

    input_num = num_input;
    hidden_layer_num = num_hidden_layer;
    hidden_num = new int[hidden_layer_num];
    for(int i=0;i<hidden_layer_num;i++)
        hidden_num[i] = num_hidden[i];
    output_num = num_output;
    learning_rate = rate;

    hidden = new Layer*[hidden_layer_num];
    hidden[0] = new Layer(input_num,num_hidden[0],learning_rate);
    for(int i=1;i<hidden_layer_num;i++)
        hidden[i] = new Layer(hidden_num[i-1],hidden_num[i],learning_rate);
    output = new Layer(hidden_num[hidden_layer_num-1],output_num,learning_rate);

    inputs = new double[input_num+1];
    hidden_outputs = new double*[hidden_layer_num];
    hidden_output_temp = new double*[hidden_layer_num];
    for(int i=0;i<hidden_layer_num;i++)
    {
        hidden_outputs[i] = new double[hidden_num[i]+1];
        hidden_output_temp[i] = new double[hidden_num[i]];
    }
    outputs = new double[output_num];

    inputs[input_num]=1;
    for(int i=0;i<hidden_layer_num;i++)
        hidden_outputs[i][hidden_num[i]]=1;

    hidden_errors = new double*[hidden_layer_num];
    for(int i=0;i<hidden_layer_num;i++)
        hidden_errors[i] = new double[hidden_num[i]];
    output_errors = new double[output_num];

    prev_hidden_weights = new double**[hidden_layer_num];
    prev_hidden_weights[0] = new double*[hidden_num[0]];
    for(int j=0;j<hidden_num[0];j++)
        prev_hidden_weights[0][j] = new double[input_num+1];
    for(int i=1;i<hidden_layer_num;i++)
    {
        prev_hidden_weights[i] = new double*[hidden_num[i]];
        for(int j=0;j<hidden_num[i];j++)
            prev_hidden_weights[i][j] = new double[hidden_num[i-1]+1];
    }

    prev_output_weights = new double*[output_num];
    for(int i=0;i<output_num;i++)
        prev_output_weights[i] = new double[hidden_num[hidden_layer_num-1]+1];
}

Network::~Network()
{
    for(int i=0;i<hidden_layer_num;i++)
        delete hidden[i];
    delete[] hidden;
    cout << "Hidden layers destroyed\n" << endl;
    delete output;
    cout << "Output layer destroyed\n" << endl;

    delete[] inputs;
    for(int i=0;i<hidden_layer_num;i++)
    {
        delete[] hidden_outputs[i];
        delete[] hidden_output_temp[i];
    }
    delete[] hidden_output_temp;
    delete[] hidden_outputs;
    delete[] outputs;

    for(int i=0;i<hidden_layer_num;i++)
        delete[] hidden_errors[i];
    delete[] hidden_errors;
    delete[] output_errors;

    for(int i=0;i<hidden_layer_num;i++)
    {
        for(int j=0;j<hidden_num[i];j++)
            delete[] prev_hidden_weights[i][j];
        delete[] prev_hidden_weights[i];
    }
    delete[] prev_hidden_weights;

    for(int i=0;i<output_num;i++)
        delete[] prev_output_weights[i];
    delete[] prev_output_weights;

    cout << "Network destroyed\n" << endl;
}

void Network::feedforward(double user_input[])
{
    for(int i=0;i<input_num;i++)
        inputs[i]=user_input[i];

    hidden[0]->feedforward(inputs);
    hidden[0]->get_outputs(hidden_output_temp[0]);
    for(int j=0;j<hidden_num[0];j++)
        hidden_outputs[0][j] = hidden_output_temp[0][j];
    for(int i=1;i<hidden_layer_num;i++)
    {
        hidden[i]->feedforward(hidden_outputs[i-1]);
        hidden[i]->get_outputs(hidden_output_temp[i]);
        for(int j=0;j<hidden_num[i];j++)
            hidden_outputs[i][j]=hidden_output_temp[i][j];
    }
    output->feedforward(hidden_outputs[hidden_layer_num-1]);
    output->get_outputs(outputs);
}

void Network::feedback(double user_target[])
{
    for(int i=0;i<output_num;i++)
    {
        output_errors[i]=outputs[i] * (1-outputs[i]) * (user_target[i]-outputs[i]);
        output->get_weights(i,prev_output_weights[i]);
    }
    output->update_weights(output_errors,outputs);

    {
        double error_weight[hidden_num[hidden_layer_num-1]];
        for(int i=0;i<hidden_num[hidden_layer_num-1];i++)
        {
            error_weight[i] = 0.0;
            for(int j=0;j<output_num;j++)
                error_weight[i]+=output_errors[j]*prev_output_weights[j][i];
        }

        for(int i=0;i<hidden_num[hidden_layer_num-1];i++)
        {
            hidden_errors[hidden_layer_num-1][i]=hidden_outputs[hidden_layer_num-1][i] * (1-hidden_outputs[hidden_layer_num-1][i]) * error_weight[i];
            hidden[hidden_layer_num-1]->get_weights(i,prev_hidden_weights[hidden_layer_num-1][i]);
        }
        hidden[hidden_layer_num-1]->update_weights(hidden_errors[hidden_layer_num-1],inputs);
    }

    for(int k=hidden_layer_num-2;k>=0;k--)
    {
        double error_weight[hidden_num[k]];
        for(int i=0;i<hidden_num[k];i++)
        {
            error_weight[i] = 0.0;
            for(int j=0;j<hidden_num[k+1];j++)
                error_weight[i]+=hidden_errors[k+1][j]*prev_hidden_weights[k+1][j][i];
        }

        for(int i=0;i<hidden_num[k];i++)
        {
            hidden_errors[k][i]=hidden_outputs[k][i] * (1-hidden_outputs[k][i]) * error_weight[i];
            hidden[k]->get_weights(i,prev_hidden_weights[k][i]);
        }
        hidden[k]->update_weights(hidden_errors[k],inputs);
    }
}

void Network::save_weights() const
{
    ofstream of("weights.txt", std::ofstream::out);

    for(int i=0;i<hidden_num[0];i++)
        for(int j=0;j<input_num;j++)
            of << prev_hidden_weights[0][i][j] << " ";

    for(int k=1;k<hidden_layer_num;k++)
        for(int i=0;i<hidden_num[k];i++)
            for(int j=0;j<input_num;j++)
                of << prev_hidden_weights[k][i][j] << " ";	

    for(int i=0;i<output_num;i++)
        for(int j=0;j<hidden_num[hidden_layer_num-1];j++)
            of << prev_output_weights[i][j] << " ";	

    of.close();
}

void Network::print_weights() const
{
    cout << "Hidden[0] --" << endl;
    for(int i=0;i<hidden_num[0];i++)
    {
        for(int j=0;j<input_num;j++)
        {
            if(j%5==0)
                cout << endl;
            cout << setw(15) << prev_hidden_weights[0][i][j] << "   ";
        }
        cout << endl;
    }

    for(int k=1;k<hidden_layer_num;k++)
    {
        cout << "\nHidden[" << k << "] --" << endl;
        for(int i=0;i<hidden_num[k];i++)
        {
            for(int j=0;j<input_num;j++)
            {
                if(j%5==0)
                    cout << endl;
                cout << setw(15) << prev_hidden_weights[k][i][j] << "   ";
            }
            cout << endl;
        }
    }

    cout << "\nOutput --" << endl;
    for(int i=0;i<output_num;i++)
    {
        for(int j=0;j<hidden_num[hidden_layer_num-1];j++)
        {
            if(j%5==0)
                cout << endl;
            cout << setw(15) << prev_output_weights[i][j] << "   ";
        }
        cout << endl;
    }

}

void Network::get_outputs(double array[]) const
{
    for(int i=0;i<output_num;i++)
        array[i]=outputs[i];
}

void Network::get_output_errors(double array[]) const
{
    for(int i=0;i<output_num;i++)
        array[i]=output_errors[i];
}

void Network::get_hidden_outputs(int layer, double array[]) const
{
    for(int i=0;i<hidden_num[layer];i++)
        array[i]=hidden_outputs[layer][i];
}

void Network::get_hidden_errors(int layer, double array[]) const
{
    for(int i=0;i<hidden_num[layer];i++)
        array[i]=hidden_errors[layer][i];
}

