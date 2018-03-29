#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iterator>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "BatchFactory.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

#define IMAGE_PATH "/notebooks/IVA_JPEG_640/" // Path of calibration data set
#define NUM_CAL_BATCHES 40 //number of calibration batches

const char* DEPLOY_FILE = "../../model/resnet18/resnet18.prototxt";
const char* MODEL_FILE = "../../model/resnet18/resnet18.caffemodel";
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_CONV_NAME = "Layer11_cov";
const char* OUTPUT_BLOB_BBOX_NAME = "Layer11_bbox";
static const int BATCH_SIZE = 4;
static const int INPUT_N = 100;
static const int INPUT_C = 3;
static const int INPUT_H = 368;
static const int INPUT_W = 640;
static const int CLASS_NUM = 3;
static const std::string CLASS_NAME[CLASS_NUM] = {"Person", "TwoWheelers", "car"};
static const float CLASS_THRESHOLD[CLASS_NUM] = {0.2f, 0.2f, 0.2f};
static const int BBOX_MAX_NUM = 32;
static std::vector<std::string> IMAGE_LIST = { "01.ppm", "02.ppm", "03.ppm", "04.ppm" };
static const float SCALER = 0.00392156f; // 1.0/255
static const int TIMING_ITERATIONS = 1;

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) override
        {
            // suppress info-level messages
            if (severity == Severity::kINFO) return;

            switch (severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }
} gLogger;

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(),
                [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n",
                    mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;

struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[INPUT_C * INPUT_H * INPUT_W];
};

struct BBOX
{
    float x1, y1, x2, y2;
    int cls;
};

struct OUTPUT_RESULT
{
    DimsCHW ConvDims;
    size_t ConvSize;
    float* ConvResult;

    DimsCHW BboxDims;
    size_t BboxSize;
    float* BboxResult;
};

std::string locateFile(const std::string& input)
{
    std::string file;
    std::vector<std::string> directories{"data/"};
    const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM& ppm, BBOX bbox)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
    auto round = [](float x)->int {return int(std::floor(x + 0.5f)); };
    for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
    {
        // bbox top border
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
        // bbox bottom border
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
    }
    for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
    {
        // bbox left border
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
        // bbox right border
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
    }
    outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
    private:
        // TODO1
        // Hints:
        //     1> Assisant class 'BatchFactory' is used to load and preprocess the
        //        calibration data set with OpenCV (resize, color reformat,
        //        substruct mean, scale and etc.). It's defined in BatchFactory.h.
        //     2> You can refer to its constructor to instance it,
        //          BatchFactory(string path, DimsNCHW dims, int numCalBatches)
        //          param_#1: File path of calibration data set (could be found
        //                    from above macro in the beginning of current file).
        //          param_#2: DimsNCHW{N, C, H, W} describes the desired input dimension of this network.
        //                    Note that (N * param_#3) should be less than total image
        //                    count 5806, e.g., a possible combination, N=100,
        //                    NUMBER_CAL_BATCHES=40 (100 * 40 < 5806).
        //          param_#3: Number of calibration batches.
        //     3> Understand the concepts of calibation batch size and number of
        //        calibration batches.
        //
        BatchFactory mBF{
            IMAGE_PATH,
            DimsNCHW{INPUT_N, INPUT_C, INPUT_H, INPUT_W},
            NUM_CAL_BATCHES };

        size_t mInputSize;
        bool mReadCache{ true };
        void* mDeviceInput{ nullptr };
        std::vector<char> mCalibrationCache;

    public:
        Int8EntropyCalibrator(bool readCache = true)
            : mReadCache(readCache)
        {
            // Input size of each calibration batch
            // TODO2
            // Hints:
            //     1> The input size is in element bias other than byte.
            //     2> The input size can be calculated with the dimension of calibration
            //        data set and the dimension is represented as {N, C, H, W} format.
            //     3> mBF.m_Dims.n() represents the value of 'N', and same manner to others.
            //
            mInputSize = mBF.m_Dims.n()*mBF.m_Dims.c()*mBF.m_Dims.h()*mBF.m_Dims.w();

            // Allocate GPU memory for storing processed calibration data
            CHECK(cudaMalloc(&mDeviceInput, mInputSize * sizeof(float)));
        }

        virtual ~Int8EntropyCalibrator()
        {
            CHECK(cudaFree(mDeviceInput));
        }

        // Return the calibration batch size
        int getBatchSize() const override
        {
            return mBF.m_Dims.n();
        }

        // Feed the calibration data into TensorRT.
        bool getBatch(void* bindings[], const char* names[], int nbBindings) override
        {
            float mean[3]{ .0f, .0f, .0f }; // also in BGR order
            float *batchBuf {nullptr};

            // Use BatchFactory to load processed calibration data
            batchBuf = mBF.loadBatch(mean, SCALER);

            // Calibration done
            if (!batchBuf)
                return false;

            // Copy batching data from CPU memory to GPU memory
            // TODO3
            // Hints:
            //      1> GPU memory is allocated in the constructor.
            //      2> CPU memory is allocated by BatchFactory.
            //      3> Copy direction flag could be either cudaMemcpyHostToDevice or
            //         cudaMemcpyDeviceToHost.
            //
            
            /* We have these definitions from above:
                    size_t mInputSize;
                    bool mReadCache{ true };
                    void* mDeviceInput{ nullptr };
                    std::vector<char> mCalibrationCache;
              */
            
            CHECK(cudaMemcpy(
                        mDeviceInput,
                        batchBuf,
                        mInputSize,
                        cudaMemcpyHostToDevice));

            assert(!strcmp(names[0], INPUT_BLOB_NAME));
            // Pass the GPU memory pointer to TensorRT for calibration.
            bindings[0] = mDeviceInput;

            return true;
        }

        // Cache the calibration result in local file to avoid repeated calibration
        // process during next time running
        void writeCalibrationCache(const void* cache, size_t length) override
        {
            std::ofstream output("CalibrationTable", std::ios::binary);
            output.write(reinterpret_cast<const char*>(cache), length);
        }

        // Read cached calibration table if it's already existing.
        const void* readCalibrationCache(size_t& length) override
        {
            mCalibrationCache.clear();
            std::ifstream input("CalibrationTable", std::ios::binary);
            input >> std::noskipws;
            if (mReadCache && input.good())
            {
                std::cout << "Skip calibrating and use the existing table" << std::endl;
                std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

            }
            length = mCalibrationCache.size();
            return length ? &mCalibrationCache[0] : nullptr;
        }
};

void caffeToGIEModel(const char* deployFile,
        const char* modelFile,
        const std::vector<std::string>& outputs,
        unsigned int maxBatchSize,
        IInt8Calibrator* calibrator,
        IHostMemory *&gieModelStream)
{
    // Create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();

    // TODO5
    // Hints:
    //     The precision mode of parser should be either FP32(DataType::kFLOAT) or
    //     FP16(DataType::kHALF).
    //     For INT8 mode, we have to set the precision mode in FP32 conventionally,
    //     and then TensorRT will convert the weight into INT8 after FP32 calibration.
    //
    const IBlobNameToTensor *blobNameToTensor = parser->parse(
            deployFile,
            modelFile,
            *network,
            DataType::kFLOAT);
    assert(blobNameToTensor != nullptr);

    // The caffe file has no notion of outputs, so we need to explicitly specify
    // which tensors the engine should generate
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Configure the batch size and maxWorkspaceSize
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);

    // TODO6
    // Hints:
    //     1. Use the following API from builder to set the running mode,
    //              void setInt8Mode(bool mode)
    //
    //     2. Use the following API from builder to set the calibrator,
    //              void setInt8Calibrator(IInt8Calibrator* calibrator)
    //
    setInt8Mode(true);
    setInt8Calibrator(calibrator);

    // Build the engine
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

void fillImageData(float* imageData)
{
    std::vector<PPM> ppms(BATCH_SIZE);

    assert(ppms.size() <= IMAGE_LIST.size());

    for (int i = 0; i < BATCH_SIZE; ++i)
        readPPMFile(IMAGE_LIST[i], ppms[i]);

    for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < BATCH_SIZE; ++i)
    {
        for (int c = 0; c < INPUT_C; ++c)
        {
            // the color image to input should be in BGR order
            for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
                imageData[i * volImg + c * volChl + j] =
                    float(ppms[i].buffer[j * INPUT_C + 2 - c]) * SCALER;
        }
    }
}

void doInference(
        IExecutionContext& context,
        int batchSize,
        float* input,
        struct OUTPUT_RESULT& output)
{
    const ICudaEngine& engine = context.getEngine();

    // Bindings means the input and output buffer pointers that
    // we pass to the engine.
    // The engine requires exactly ICudaEngine::getNbBindings() of these,
    // but in this case we know that there is exactly one input and two outputs.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input
    // and output tensors.
    // Note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    DimsCHW inputDims =
        static_cast<DimsCHW&&>(engine.getBindingDimensions(inputIndex));
    size_t inputSize =
        batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);

    // Output Conv Dimension and size
    int outputConvIndex= engine.getBindingIndex(OUTPUT_BLOB_CONV_NAME);
    output.ConvDims =
        static_cast<DimsCHW&&>(engine.getBindingDimensions(outputConvIndex));
    output.ConvSize =
        batchSize * output.ConvDims.c() * output.ConvDims.h() * output.ConvDims.w() * sizeof(float);

    // Output Bbox Dimension and size
    int outputBboxIndex = engine.getBindingIndex(OUTPUT_BLOB_BBOX_NAME);
    output.BboxDims =
        static_cast<DimsCHW&&>(engine.getBindingDimensions(outputBboxIndex));
    output.BboxSize =
        batchSize * output.BboxDims.c() * output.BboxDims.h() * output.BboxDims.w() * sizeof(float);

    // Allocate GPU buffers
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputConvIndex], output.ConvSize));
    CHECK(cudaMalloc(&buffers[outputBboxIndex], output.BboxSize));

    // Copy the input buffer
    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize,
                cudaMemcpyHostToDevice));

    // Do real inference on the input data
    for (int i = 0; i < TIMING_ITERATIONS;i++)
        context.execute(batchSize, buffers);

    // Allocate CPU buffers to store the result
    output.ConvResult = new float[output.ConvSize];
    output.BboxResult = new float[output.BboxSize];

    CHECK(cudaMemcpy(output.ConvResult, buffers[outputConvIndex], output.ConvSize,
                cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(output.BboxResult, buffers[outputBboxIndex], output.BboxSize,
                cudaMemcpyDeviceToHost));

    // Release the GPU buffers
    // CPU Buffer will be released after bbox parsing done
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputConvIndex]));
    CHECK(cudaFree(buffers[outputBboxIndex]));
}

void parseResult(struct OUTPUT_RESULT& output)
{
    DimsCHW convDims = output.ConvDims;
    DimsCHW bboxDims = output.BboxDims;

    assert(CLASS_NUM == convDims.c());

    int grid_x_ = convDims.w();
    int grid_y_ = convDims.h();
    int gridsize_ = grid_x_ * grid_y_;
    int target_shape[2] = {grid_x_, grid_y_};
    float bbox_norm[2] = {640.0, 368.0};
    float gc_centers_0[target_shape[0]];
    float gc_centers_1[target_shape[1]];
    for (int i = 0; i < target_shape[0]; i++)
    {
        gc_centers_0[i] = (float)(i * 16 + 0.5);
        gc_centers_0[i] /= (float)bbox_norm[0];

    }
    for (int i = 0; i < target_shape[1]; i++)
    {
        gc_centers_1[i] = (float)(i * 16 + 0.5);
        gc_centers_1[i] /= (float)bbox_norm[1];
    }

    std::ofstream bbox_result;
    bbox_result.open("bbox_result.txt", std::ofstream::out | std::ofstream::trunc);

    for (int i = 0; i < BATCH_SIZE; i++)
    {
        std::vector<cv::Rect> *rectListClass;
        rectListClass = new std::vector<cv::Rect>[CLASS_NUM];
        BBOX bbox[BBOX_MAX_NUM];
        int bbox_num = 0;
        float* convPtr = output.ConvResult + i * convDims.c() * convDims.h() * convDims.w();
        float* bboxPtr = output.BboxResult + i * bboxDims.c() * bboxDims.h() * bboxDims.w();
        for (int c = 0; c < CLASS_NUM; c++)
        {
            const float *output_x1 = bboxPtr + c * (bboxDims.c()/CLASS_NUM) * bboxDims.h() * bboxDims.w();
            const float *output_y1 = output_x1 + bboxDims.h() * bboxDims.w();
            const float *output_x2 = output_y1 + bboxDims.h() * bboxDims.w();
            const float *output_y2 = output_x2 + bboxDims.h() * bboxDims.w();
            for (int h = 0; h < grid_y_; h++)
            {
                for (int w = 0; w < grid_x_; w++)
                {
                    int j = w + h * grid_x_;
                    if (convPtr[c * gridsize_ + j] >= CLASS_THRESHOLD[c])
                    {
                        float rectx1_f, recty1_f, rectx2_f, recty2_f;
                        int rectx1, recty1, rectx2, recty2;

                        rectx1_f = output_x1[w + h * grid_x_] - gc_centers_0[w];
                        recty1_f = output_y1[w + h * grid_x_] - gc_centers_1[h];
                        rectx2_f = output_x2[w + h * grid_x_] + gc_centers_0[w];
                        recty2_f = output_y2[w + h * grid_x_] + gc_centers_1[h];

                        rectx1_f *= (float)(-bbox_norm[0]);
                        recty1_f *= (float)(-bbox_norm[1]);
                        rectx2_f *= (float)(bbox_norm[0]);
                        recty2_f *= (float)(bbox_norm[1]);

                        rectx1 = (int)rectx1_f;
                        recty1 = (int)recty1_f;
                        rectx2 = (int)rectx2_f;
                        recty2 = (int)recty2_f;

                        rectx1 = rectx1 < 0 ? 0 :
                            (rectx1 >= INPUT_W ? (INPUT_W - 1) : rectx1);
                        rectx2 = rectx2 < 0 ? 0 :
                            (rectx2 >= INPUT_W ? (INPUT_W - 1) : rectx2);
                        recty1 = recty1 < 0 ? 0 :
                            (recty1 >= INPUT_H ? (INPUT_H - 1) : recty1);
                        recty2 = recty2 < 0 ? 0 :
                            (recty2 >= INPUT_H ? (INPUT_H - 1) : recty2);

                        rectListClass[c].push_back(cv::Rect(rectx1, recty1, rectx2-rectx1, recty2-recty1));
                    }
                }
            }
            cv::groupRectangles(rectListClass[c], 1, 0.2);

            std::vector<cv::Rect>& rectList = rectListClass[c];
            for (int iRect = 0; iRect < (int)rectList.size(); ++iRect)
            {
                cv::Rect& r = rectList[iRect];
                bbox[bbox_num].x1 = (float)r.x;
                bbox[bbox_num].y1= (float)r.y;
                bbox[bbox_num].x2 = bbox[bbox_num].x1 + (float)r.width;
                bbox[bbox_num].y2 = bbox[bbox_num].y1 + (float)r.height;
                bbox[bbox_num].cls = c;
                bbox_num++;
           }
        }

        // If there is bbox detected, we draw it on the original image
        if (bbox_num)
        {
            PPM ppm;
            readPPMFile(IMAGE_LIST[i], ppm);
            std::string storeName = IMAGE_LIST[i].substr(0, 2) + "_bboxed.ppm";

            for (int k = 0; k < bbox_num; k++)
            {
                std::cout << "Drawing bbox (" << CLASS_NAME[bbox[k].cls] <<
                    "," << bbox[k].x1 <<
                    "," << bbox[k].y1 <<
                    "," << bbox[k].x2 <<
                    "," << bbox[k].y2 <<
                    ") " << "on the " << IMAGE_LIST[i] << std::endl;
                writePPMFileWithBBox(storeName, ppm, bbox[k]);
                if (bbox_result.is_open())
                {
                    bbox_result <<
                        i << "," <<
                        bbox[k].x1 << "," <<
                        bbox[k].y1 << "," <<
                        bbox[k].x2 << "," <<
                        bbox[k].y2 << "\n";
                }
            }
        }
        delete [] rectListClass;
    }
    bbox_result.close();
}

int main(int argc, char** argv)
{
    IHostMemory *gieModelStream{nullptr};

    ////////////////////////////////////////////////////////////////////
    // <0> Instantiate an INT8 Calibrator
    ////////////////////////////////////////////////////////////////////
    Int8EntropyCalibrator int8Calibrator{true};

    ////////////////////////////////////////////////////////////////////
    // <1> Parse the caffe model into TensorRT recognized stream
    ////////////////////////////////////////////////////////////////////
    //
    // TODO4
    caffeToGIEModel(
            DEPLOY_FILE,
            MODEL_FILE,
            std::vector<std::string>{OUTPUT_BLOB_CONV_NAME, OUTPUT_BLOB_BBOX_NAME},
            BATCH_SIZE,
            &int8Calibrator;
            gieModelStream);

    ////////////////////////////////////////////////////////////////////
    // <2> Deserialize the model stream and create an execution context
    ////////////////////////////////////////////////////////////////////
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(
            gieModelStream->data(),
            gieModelStream->size(),
            nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    // Allocate resource for input image and output result
    float* inputData = new float[BATCH_SIZE * INPUT_C * INPUT_H * INPUT_W];
    struct OUTPUT_RESULT outputData;

    ////////////////////////////////////////////////////////////////////
    // <3> Prepare image data
    ////////////////////////////////////////////////////////////////////
    fillImageData(inputData);

    ////////////////////////////////////////////////////////////////////
    // <4> Run real inference on the image
    ////////////////////////////////////////////////////////////////////
    doInference(*context, BATCH_SIZE, inputData, outputData);

    ////////////////////////////////////////////////////////////////////
    // <5> Parse the output result
    ////////////////////////////////////////////////////////////////////
    parseResult(outputData);

    // Destroy the runtime
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Release the input and output CPU buffers
    delete[] inputData;
    delete[] outputData.ConvResult;
    delete[] outputData.BboxResult;

    // Print the execution time
    //gProfiler.printLayerTimes();

    return 0;
}

