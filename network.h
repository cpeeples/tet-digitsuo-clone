#ifndef NETWORK_H
#define NETWORK_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64

typedef struct {
    float *hidden_weights;
    float *hidden_bias;
    float *output_weights;
    float *output_bias;
    float *hidden_weights_momentum;
    float *hidden_bias_momentum;
    float *output_weights_momentum;
    float *output_bias_momentum;
} Network;

#endif