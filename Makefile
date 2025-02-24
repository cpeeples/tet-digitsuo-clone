CC = gcc
CFLAGS = -Wall -Wextra -O2 -Wunused -Wuninitialized -Wshadow
LDFLAGS = -lncurses -lm

NVCC = nvcc
CUDA_CFLAGS = -O3 -arch=sm_86  # RTX 3090 Compute Capability 8.6
CUDA_LDFLAGS = -lcudart -lgomp -lm -lz

SRC_DIR = src
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/draw_interface.c $(SRC_DIR)/neural_net.c $(SRC_DIR)/utils.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = digit_recognition

TRAIN_SRC = train.c
TRAIN_OBJ = $(TRAIN_SRC:.c=.o)
TRAIN_CUDA_SRC = cuda_kernels.cu
TRAIN_CUDA_OBJ = $(TRAIN_CUDA_SRC:.cu=.o)
TRAIN_TARGET = train
TRAIN_FLAGS = -Wall -Wextra -O3 -march=native -Wunused -Wuninitialized -Wshadow -fopenmp
TRAIN_LIBS = -lm -lz -fopenmp

.PHONY: all clean delete_debug train train-cuda doxygen

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

train: $(TRAIN_SRC)
	$(CC) $(TRAIN_FLAGS) $(TRAIN_SRC) -o $(TRAIN_TARGET) $(TRAIN_LIBS)

train-cuda: $(TRAIN_OBJ) $(TRAIN_CUDA_OBJ)
	$(NVCC) $(CUDA_CFLAGS) $(TRAIN_OBJ) $(TRAIN_CUDA_OBJ) -o $(TRAIN_TARGET) $(CUDA_LDFLAGS)

$(TRAIN_OBJ): $(TRAIN_SRC)
	$(CC) $(TRAIN_FLAGS) -c $(TRAIN_SRC) -o $(TRAIN_OBJ)

$(TRAIN_CUDA_OBJ): $(TRAIN_CUDA_SRC)
	$(NVCC) $(CUDA_CFLAGS) -c $(TRAIN_CUDA_SRC) -o $(TRAIN_CUDA_OBJ)

docs:
	@command -v doxygen >/dev/null 2>&1 || { echo "Error: doxygen is not installed. Please install it first."; exit 1; }
	@echo "Generating HTML documentation..."
	@doxygen Doxyfile
	@echo "Documentation generated in docs/html/"

clean:
	rm -f $(OBJECTS) $(TARGET) $(TRAIN_TARGET) $(TRAIN_OBJ) $(TRAIN_CUDA_OBJ) debug.log

clean_docs:
	rm -rf docs

clean_debug:
	rm -f debug.log