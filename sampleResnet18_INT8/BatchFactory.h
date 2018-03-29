#include <iostream>
#include <vector>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NvInfer.h"

using namespace std;
using namespace nvinfer1;

class BatchFactory
{
public:
    DimsNCHW m_Dims;

    BatchFactory(string path, DimsNCHW dims, int numCalBatches)
        : m_Dims(dims), m_imgCounts(0), m_numCalBatches(numCalBatches)
    {
        // Find all images under the path
        cv::glob(path, m_fileNames, false);

        // The calibration dataset should be less than the image counts
        assert(dims.n() * m_numCalBatches <= m_fileNames.size());

        cout << "There are " << m_fileNames.size() << " images in total"<< endl;

        // Allocate float type CPU memory for a batch of image
        m_imgBuf = (float *) malloc(m_Dims.n() * m_Dims.c() * m_Dims.h() * m_Dims.w() * sizeof(float));
    }

    ~BatchFactory()
    {
        free(m_imgBuf);
    }

    float *loadBatch(float *mean, float scaler)
    {
        // m_imgCounts records how many images have been loaded already
        if(m_imgCounts == m_Dims.n() * m_numCalBatches)
        {
            cout << "Complete loading calibration image " << endl;
            return NULL;
        }

        assert(mean != NULL && scaler != .0f);

        cout << "Trying to load calibration image " <<
            m_imgCounts << " ~ " << m_imgCounts + m_Dims.n()  << endl;

        for (unsigned i = m_imgCounts, j = 0; i < m_imgCounts + m_Dims.n(); i++, j++)
        {
            cv::Mat src;
            src = cv::imread(m_fileNames[i], CV_LOAD_IMAGE_COLOR);
            if (!src.data)
            {
                cout << "Could not open or find the image: " << m_fileNames[i] << endl;
                return NULL;
            }

            // Resize as requested by user
            cv::Mat dst;
            cv::Size size(m_Dims.w(), m_Dims.h());
            cv::resize(src, dst, size);

            // Split channels
            cv::Mat bgr[3];
            cv::split(dst, bgr);

            size_t volBatch = m_Dims.c() * m_Dims.h() * m_Dims.w();
            size_t volCh = m_Dims.h() * m_Dims.w();

            // Convert to float, sub by mean and mul by scaler
            for (unsigned c = 0; c < m_Dims.c(); c++)
                for (unsigned y = 0; y < m_Dims.h(); y++)
                    for (unsigned x = 0; x < m_Dims.w(); x++)
                    {
                        m_imgBuf[j * volBatch + c * volCh + y * m_Dims.w() + x] =
                            ((float)bgr[c].data[y * m_Dims.w() + x] - mean[c]) * scaler;
                    }
        }

        m_imgCounts += m_Dims.n();

        return m_imgBuf;
    }

private:
    vector<string> m_fileNames;
    float *m_imgBuf {nullptr};
    unsigned m_imgCounts;
    unsigned m_numCalBatches;
};

