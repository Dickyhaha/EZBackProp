#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sys/time.h>

#include "BackPropagation/network.h"

using namespace std;

struct Frame
{
    int index;
    double image[784];
};

int parse(vector<Frame>& fr)
{
    unsigned long label_size, image_size;
    char* label_buffer;
    char* image_buffer;

    ifstream label("../datasets/mnist/train-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream image("../datasets/mnist/train-images.idx3-ubyte", ios::in | ios::binary);
    if(!label.is_open() || !image.is_open())
        return -1;


    label.seekg(0,ios::end);
    label_size = label.tellg();
    label_buffer = new char[label_size];
    label.seekg(0,ios::beg);
    label.read(label_buffer,label_size);
    label.close();

    image.seekg(0,ios::end);
    image_size = image.tellg();
    image_buffer = new char[image_size];
    image.seekg(0,ios::beg);
    image.read(image_buffer,image_size);
    image.close();

    for(int k=0;k<60000;k++)
    {
        fr[k].index = label_buffer[k+8];
        for(int i=0;i<28;i++)
            for(int j=0;j<28;j++)
                fr[k].image[i*28+j] = static_cast<double>(image_buffer[k*784 + i*28 + j + 16]/255.0);
        //cout << fr[k].image << endl;
    }

    delete image_buffer;
    delete label_buffer;

    cout << "Bin parsed" << endl;
    cout << "Size: " << image_size << endl;

    return 0;
}

int main()
{
    vector<Frame> frame;
    frame.resize(60000);

    if(parse(frame) == -1)
    {
        cerr << "Error while parsing!" << endl;
        return 0;
    }

    int layer[] = {300};
    //int layer[] = {500, 250, 100};
    Network net(784, 1, layer, 10, 0.1);

    //vector<int> num;
    //num << 784 << 500 << 10;
    //Network<float> net(num, 0.1f);

    double input[784];
    double output[10];
    double target;

    //vector<float> input, output;
    //input.resize(784);
    //output.resize(10);

    int error_count=0;
    int count = 0;
    int round = 0;
    int max;

    timeval start,current;
    gettimeofday(&start,NULL);

    while(true)
    {

        for(int i=0;i<28;i++)
            for(int j=0;j<28;j++)
                input[i*28+j] = frame[count%60000].image[i*28+j]!=0.0;

        net.feedforward(input);
        net.get_outputs(output);
        target = frame[count%60000].index;
        
        max = 0;
        for(int i=0;i<10;i++)
            if(output[i]>output[max])
                max = i;

        if(max!=target)
            error_count++;

        gettimeofday(&current,NULL);
        if( (current.tv_sec-start.tv_sec)*1000 + (current.tv_usec-start.tv_usec)/1000 > 1000 )
        {
            gettimeofday(&start,NULL);

            for(int i=0;i<28;i++)
            {
                for(int j=0;j<28;j++)
                    cout << ((frame[count%60000].image[i*28+j])?"x ":"  ");
                cout << endl;
            }

            cout << "\n\n";
            cout << "Count: " << count << endl;
            cout << "Round: " << round << endl;
            cout << "Input index: " << count%60000 << endl;
            cout << "Target: " << target << endl;
            cout << "Output: " << max << "--" << output[max] << endl;
            cout << "Error count: " << error_count << endl;
            cout << "Error rate: " << 1.0*error_count/round << endl;
            cout << endl;
            for(int i=0;i<10;i++)
                cout << output[i] << "\t";
            cout << "\n\n\n\n\n";

            round = error_count = 0;    
        }

        for(int i=0;i<10;i++)
            output[i] = (i==target);
        net.feedback(output);

        round++;
        count++;

    }

    return 0;
}
