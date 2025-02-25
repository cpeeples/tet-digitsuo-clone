#include <cuda_runtime.h>
#include "network.h"

void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) if (input[i] > max_val) max_val = input[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) output[i] /= sum;
}

__global__ void forward_hidden_kernel(const float *batch_X, const float *hidden_weights, const float *hidden_bias,
                                      float *hidden_layer, int batch_size, int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < batch_size && j < hidden_size) {
        float sum = hidden_bias[j];
        for (int k = 0; k < input_size; k++) {
            sum += batch_X[i * input_size + k] * hidden_weights[k * hidden_size + j];
        }
        hidden_layer[i * hidden_size + j] = (sum > 0) ? sum : 0;
    }
}

__global__ void backward_dw_hidden_kernel(const float *batch_X, const float *hidden_error, float *dw_hidden,
                                          int batch_size, int input_size, int hidden_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < input_size && k < hidden_size) {
        float grad = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            grad += batch_X[i * input_size + j] * hidden_error[i * hidden_size + k];
        }
        dw_hidden[j * hidden_size + k] = grad / batch_size;
    }
}

extern "C" void forward_pass_cuda(const Network *net, const float *batch_X, float *hidden_layer, float *output_layer) {
    float *d_batch_X, *d_hidden_weights, *d_hidden_bias, *d_hidden_layer, *d_output_weights, *d_output_bias, *d_output_layer;
    cudaMalloc(&d_batch_X, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_bias, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_layer, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_bias, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_layer, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(d_batch_X, batch_X, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_weights, net->hidden_weights, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_bias, net->hidden_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, net->output_weights, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias, net->output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((BATCH_SIZE + 15) / 16, (HIDDEN_SIZE + 15) / 16);
    forward_hidden_kernel<<<blocks, threads>>>(d_batch_X, d_hidden_weights, d_hidden_bias, d_hidden_layer,
                                               BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(hidden_layer, d_hidden_layer, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < BATCH_SIZE; i++) {
        float tmp[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float sum = net->output_bias[j];
            for (int k = 0; k < HIDDEN_SIZE; k++) {
                sum += hidden_layer[i * HIDDEN_SIZE + k] * net->output_weights[k * OUTPUT_SIZE + j];
            }
            tmp[j] = sum;
        }
        softmax(tmp, &output_layer[i * OUTPUT_SIZE], OUTPUT_SIZE);
    }

    cudaFree(d_batch_X); cudaFree(d_hidden_weights); cudaFree(d_hidden_bias); cudaFree(d_hidden_layer);
    cudaFree(d_output_weights); cudaFree(d_output_bias); cudaFree(d_output_layer);
}