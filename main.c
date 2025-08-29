#include "./matrix.h"
#include "./network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NumInputNeurons 784
#define NumOutputNeurons 10
#define NumHiddenNeurons 300
#define NumHiddenLayers 4

#define NumEpochs 1
#define trainingsetsize 4
#define learningrate 0.00001



data_set * load_from_csv(char * path);
static int argmax_one_hot( matrix *y);
static void print_img28x28( matrix *x);

int main(int argc, char **argv){
    if (argc == 1) {
        network * net = net_create(NumInputNeurons, NumOutputNeurons, NumHiddenNeurons, NumHiddenLayers);
        data_set * MNIST = load_from_csv("./data/archive (1)/mnist_train.csv");
        if (MNIST && MNIST->size > 0) {
            printf("First sample label: %d\n", argmax_one_hot(MNIST->entry[0].output));
            print_img28x28(MNIST->entry[0].input);
        }
        net_train(net, NumEpochs, MNIST, learningrate);
        return 0;
    } else if (argc == 2) {
        // TODO: loading net from file path argv[1]
        fprintf(stdout, "model loading from '%s' not implemented yet\n", argv[1]);
        return 0;
    } else {
        fprintf(stderr, "expected 0 or 1 argument but got %d\n", argc - 1);
        return 1;
    }
}

data_set * load_from_csv(char * path){
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("failed to open CSV");
        return NULL;
    }
    const size_t BUFSZ = 65536;
    char *line = (char*)malloc(BUFSZ);
    if (!line) { fclose(f); perror("malloc"); return NULL; }

    if (!fgets(line, BUFSZ, f)) { free(line); fclose(f); return NULL; }
    int count = 0;
    while (fgets(line, BUFSZ, f)) {
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r') continue;
        count++;
    }

    data_set *set = (data_set*)malloc(sizeof(data_set));
    if (!set) { free(line);fclose(f);perror("malloc set"); return NULL; }
    set->size = count;
    set->entry = (data_point*)malloc(sizeof(data_point)* (size_t)count);
    if (!set->entry) { free(set); free(line); fclose(f); perror("malloc entries"); return NULL; }

    rewind(f);
    (void)fgets(line, BUFSZ, f);

    int idx = 0;
    while (idx < count && fgets(line, BUFSZ, f)) {
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '\r') continue;
        char *token = strtok(line, ",\n\r");
        if (!token) continue; 
        int label = (int)strtol(token, NULL, 10);

        matrix *input= matrix_create(NumInputNeurons, 1, NULL);
        int pix = 0;
        while (pix < NumInputNeurons) {
            token = strtok(NULL, ",\n\r");
            if (!token) break;
            float v = strtof(token,NULL);
            input->entries[pix++] = v / 255.0f;
        }
        for (; pix < NumInputNeurons; ++pix)input->entries[pix] = 0.0f;
        matrix *output = matrix_create(NumOutputNeurons, 1, NULL);
        for (int k = 0; k < NumOutputNeurons; ++k)output->entries[k] = 0.0f;
        if (label >= 0 && label < NumOutputNeurons) output->entries[label] = 1.0f;
        set->entry[idx].input =input;
        set->entry[idx].output = output;
        idx++;
    }
    set->size = idx;

    free(line);
    fclose(f);
    return set;
}

static int argmax_one_hot(matrix *Y) {
    int best = 0;
    float bv = Y->entries[0];
    for (int i = 1; i < Y->rows * Y->cols; ++i) {
        if (Y->entries[i] > bv) { 
            bv = Y->entries[i]; 
            best = i; 
        }
    }
    return best;
}

static void print_img28x28(matrix *X) {
    static const char shades[] = " .:-=+*#%%@";
    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            float v = X->entries[r * 28 + c];
            if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
            int idx = (int)(v * 9.0f + 0.5f);
            if (idx < 0) idx = 0; if (idx > 9) idx = 9;
            putchar(shades[idx]);
        }
        putchar('\n');
    }
}

