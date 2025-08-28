#include "./matrix.h"
#include "./network.h"
#include <stdio.h>
#include <stdlib.h>

#define NumInputNeurons 2
#define NumOutputNeurons 2
#define NumHiddenNeurons 2
#define NumHiddenLayers 1

#define NumEpochs 100000
#define trainingsetsize 4
#define learningrate 0.01  // Reduzierte Lernrate



int main(){
    setvbuf(stdout, NULL, _IONBF, 0);
    network * net = net_create(NumInputNeurons, NumOutputNeurons, NumHiddenNeurons, NumHiddenLayers);
    printf("Network structure:\n");
    //net_print(net);
    
    net->learning_rate = learningrate;

    float training_inputs[4][NumInputNeurons] = {{0.0,0.0},
                                                  {1.0,0.0},
                                                  {0.0,1.0},
                                                  {1.0,1.0}};
    
    float training_outputs[4][NumOutputNeurons] = {{1.0, 0.0},
                                                   {0.0, 1.0}, 
                                                   {0.0, 1.0}, 
                                                   {1.0, 0.0}}; 

    data_set * set = malloc(sizeof(data_set));
    set->size = trainingsetsize;
    set->entry = malloc(trainingsetsize * sizeof(data_point));

    for (int i = 0; i < trainingsetsize; i++){
        set->entry[i].input = matrix_create(NumInputNeurons, 1, training_inputs[i]);
        set->entry[i].output = matrix_create(NumOutputNeurons, 1, training_outputs[i]);
    }

    //net_train(net, NumEpochs, set, learningrate);
    //for(int i = 0; i < NumEpochs; i++) test(net, set);
    net_train(net, NumEpochs, set, learningrate);
    

    for(int i = 0; i < trainingsetsize; i++){
        matrix ** A = NULL;
        matrix * out = forward(net, set->entry[i].input, &A);
        printf("input:\n\n");
        matrix_print(set->entry[i].input);
        printf("output:\n\n");
        matrix_print(out);
        printf("\n\n");
        
    }
    //float acc = accuracy(net, set);
    //printf("%f", acc);
    
    for (int i = 0; i < trainingsetsize; i++) {
        matrix_free(set->entry[i].input);
        matrix_free(set->entry[i].output);
    }
    free(set->entry);
    free(set);
    free(net->layers);
    free(net);
    
    return 0;
}