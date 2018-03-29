APP := sample_resnet18_int8

CC := g++

CUDA_INSTALL_PATH ?= /usr/local/cuda
TRT_INSTALL_PATH ?= /tensorrt/TensorRT-3.0.1

SRCS := \
	sampleResnet18_INT8.cpp

OBJS := $(SRCS:.cpp=.o)

CPPFLAGS := \
	-std=c++11\
	-I"$(TRT_INSTALL_PATH)/include" \
	-I"$(CUDA_INSTALL_PATH)/include"

LDFLAGS := \
	-lnvparsers \
	-lnvinfer \
	-lcudart \
	-lopencv_objdetect -lopencv_highgui -lopencv_imgproc -lopencv_core \
	-L"$(TRT_INSTALL_PATH)/lib" \
	-L"$(CUDA_INSTALL_PATH)/lib64"

all: $(APP)

%.o: %.cpp
	@echo "Compiling: $<"
	@ $(CC) $(CPPFLAGS) -c $<

$(APP): $(OBJS)
	@echo "Linking: $@"
	@ $(CC) -o $@ $(OBJS) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -rf $(APP) $(OBJS) *.ppm *.txt                   

