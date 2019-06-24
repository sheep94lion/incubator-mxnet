#include <string>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include "mxnet/c_predict_api.h"

const mx_float DEFAULT_MEAN = 117.0;

// Read file to buffer
class BufferFile {
    public :
        std::string file_path_;
        std::size_t length_ = 0;
        std::unique_ptr<char[]> buffer_;

        explicit BufferFile(const std::string& file_path)
            : file_path_(file_path) {

            std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);


            ifs.seekg(0, std::ios::end);
            length_ = static_cast<std::size_t>(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

            // Buffer as null terminated to be converted to string
            buffer_.reset(new char[length_ + 1]);
            buffer_[length_] = 0;
            ifs.read(buffer_.get(), length_);
            ifs.close();
        }

        std::size_t GetLength() {
            return length_;
        }

        char* GetBuffer() {
            return buffer_.get();
        }
};

void GetImageFile(const std::string& image_file,
                  mx_float* image_data, int channels,
                  cv::Size resize_size, const mx_float* mean_data = nullptr) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, cv::IMREAD_COLOR);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    int size = im.rows * im.cols * channels;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    float mean_b, mean_g, mean_r;
    mean_b = mean_g = mean_r = DEFAULT_MEAN;

    for (int i = 0; i < im.rows; i++) {
        auto data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            if (mean_data) {
                mean_r = *mean_data;
                if (channels > 1) {
                    mean_g = *(mean_data + size / 3);
                    mean_b = *(mean_data + size / 3 * 2);
                }
                mean_data++;
            }
            if (channels > 1) {
                *ptr_image_b++ = static_cast<mx_float>(*data++) - mean_b;
                *ptr_image_g++ = static_cast<mx_float>(*data++) - mean_g;
            }

            *ptr_image_r++ = static_cast<mx_float>(*data++) - mean_r;;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(const std::string& synset_file) {
    std::ifstream fi(synset_file.c_str());

    if (!fi.is_open()) {
        std::cerr << "Error opening synset file " << synset_file << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while (fi >> synset) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

static std::string trim(const std::string& input) {
    auto not_space = [](int ch) {
        return !std::isspace(ch);
    };
    auto output = input;
    output.erase(output.begin(), std::find_if(output.begin(), output.end(), not_space));
    output.erase(std::find_if(output.rbegin(), output.rend(), not_space).base(), output.end());
    return output;
}

void PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size do not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    std::size_t best_idx = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        // LOGI("Accuracy[%d] = %.8f", i, data[i]);

        if (data[i] > best_accuracy) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }

}

void predict(PredictorHandle pred_hnd, const std::vector<mx_float> &image_data,
             const std::string &synset_file, int i) {
    auto image_size = image_data.size();
    // Set Input Image
    MXPredSetInput(pred_hnd, "data", image_data.data(), static_cast<mx_uint>(image_size));

    // Do Predict Forward
    MXPredForward(pred_hnd);

    mx_uint output_index = 0;

    mx_uint* shape = nullptr;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(pred_hnd, output_index, &shape, &shape_len);

    std::size_t size = 1;
    for (mx_uint i = 0; i < shape_len; ++i) { size *= shape[i]; }

    std::vector<float> data(size);

    MXPredGetOutput(pred_hnd, output_index, &(data[0]), static_cast<mx_uint>(size));

    // Release Predictor
    MXPredFree(pred_hnd);

    // Synset path for your model, you have to modify it
    auto synset = LoadSynset(synset_file);

    // Print Output Data
    PrintOutputResult(data, synset);
}

int main() {
    std::string hello = "Hello from C++, load file";
    std::string json_file = "/home/yizhao/Code/saved_model/ddff/dff_cur_sym.json";
    std::string param_file = "/home/yizhao/Code/saved_model/ddff/rfcn_dff_flownet_vid-0000.params";
    std::string synset_file = "/home/yizhao/Code/saved_model/ddff/synset.txt";

    BufferFile json_data(json_file);
    BufferFile param_data(param_file);

    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 4;  // 1 for feedforward
    const char* input_key[4] = { "data", "im_info", "data_key", "feat_key"};
    const char** input_keys = input_key;
    // Image size and channels
    int width = 1000;
    int height = 562;
    int channels = 3;

    const mx_uint input_shape_indptr[5] = { 0, 4, 6, 10, 14};
    const mx_uint input_shape_data[14] = { 1,
                                          static_cast<mx_uint>(channels),
                                          static_cast<mx_uint>(height),
                                          static_cast<mx_uint>(width),
                                          1,
                                          3,
                                          1,
                                          static_cast<mx_uint>(channels),
                                          static_cast<mx_uint>(height),
                                          static_cast<mx_uint>(width),
                                          1,
                                          1024,
                                          1,
                                          1};

    PredictorHandle pred_hnd;
    printf("PredictorHandle: %x\n", pred_hnd);
    int suc = MXPredCreate(static_cast<const char*>(json_data.GetBuffer()),
                 static_cast<const char*>(param_data.GetBuffer()),
                 static_cast<int>(param_data.GetLength()),
                 dev_type,
                 dev_id,
                 num_input_nodes,
                 input_keys,
                 input_shape_indptr,
                 input_shape_data,
                 &pred_hnd);
    // assert(pred_hnd);
    printf("Success? %d\n", suc);
    printf("PredictorHandle: %x\n", pred_hnd);
    auto image_size = static_cast<std::size_t>(width * height * channels);
    std::vector<mx_float> image_data(image_size);
    std::string test_file = "/home/yizhao/Code/saved_model/cat.jpg";
    GetImageFile(test_file, image_data.data(), channels, cv::Size(width, height));

    predict(pred_hnd, image_data, synset_file, 0);
}
