#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T>;

template <typename T>
UniquePtr<T> make_unique_trt(T* ptr) {
    return UniquePtr<T>(ptr);
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

std::vector<char> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), static_cast<std::streamsize>(size));

    if (!file) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return buffer;
}

std::vector<std::string> load_labels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file) {
        throw std::runtime_error("Failed to open labels file: " + labels_path);
    }

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }

    return labels;
}

cv::Mat resize_short_side(const cv::Mat& image_rgb, int short_side = 232) {
    int h = image_rgb.rows;
    int w = image_rgb.cols;

    int new_h = 0;
    int new_w = 0;

    if (h < w) {
        new_h = short_side;
        new_w = static_cast<int>(std::round(w * static_cast<float>(short_side) / h));
    } else {
        new_w = short_side;
        new_h = static_cast<int>(std::round(h * static_cast<float>(short_side) / w));
    }

    cv::Mat resized;
    cv::resize(image_rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    return resized;
}

cv::Mat center_crop(const cv::Mat& image_rgb, int crop_size = 224) {
    int h = image_rgb.rows;
    int w = image_rgb.cols;

    int top = (h - crop_size) / 2;
    int left = (w - crop_size) / 2;

    if (top < 0 || left < 0 || top + crop_size > h || left + crop_size > w) {
        throw std::runtime_error("Center crop failed due to invalid size.");
    }

    cv::Rect roi(left, top, crop_size, crop_size);
    return image_rgb(roi).clone();
}

std::vector<float> preprocess_image(const std::string& image_path) {
    cv::Mat image_bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image_bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path);
    }

    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

    image_rgb = resize_short_side(image_rgb, 232);
    image_rgb = center_crop(image_rgb, 224);

    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0 / 255.0);

    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    std::vector<cv::Mat> channels(3);
    cv::split(image_rgb, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / stdv[c];
    }

    std::vector<float> chw(3 * 224 * 224);
    int channel_size = 224 * 224;

    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            chw.data() + c * channel_size,
            channels[c].data,
            channel_size * sizeof(float)
        );
    }

    return chw;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());

    std::vector<float> exps(logits.size());
    float sum = 0.0f;

    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i] - max_val);
        sum += exps[i];
    }

    for (auto& v : exps) {
        v /= sum;
    }

    return exps;
}

std::vector<int> topk_indices(const std::vector<float>& probs, int k = 5) {
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(
        indices.begin(),
        indices.begin() + k,
        indices.end(),
        [&probs](int a, int b) { return probs[a] > probs[b]; }
    );

    indices.resize(k);
    return indices;
}

void check_cuda(cudaError_t err, const std::string& msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0]
                      << " <engine.plan> <imagenet_classes.txt> <image1> [image2] [image3] ..."
                      << std::endl;
            return 1;
        }

        std::string engine_path = argv[1];
        std::string labels_path = argv[2];

        std::vector<std::string> image_paths;
        for (int i = 3; i < argc; ++i) {
            image_paths.push_back(argv[i]);
        }

        if (!file_exists(engine_path)) {
            throw std::runtime_error("Engine file not found: " + engine_path);
        }

        if (!file_exists(labels_path)) {
            throw std::runtime_error("Labels file not found: " + labels_path);
        }

        auto labels = load_labels(labels_path);

        Logger logger;

        auto engine_data = read_binary_file(engine_path);

        auto runtime = make_unique_trt(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime.");
        }

        auto engine = make_unique_trt(
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
        );
        if (!engine) {
            throw std::runtime_error("Failed to deserialize TensorRT engine.");
        }

        auto context = make_unique_trt(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create execution context.");
        }

        std::string input_name;
        std::string output_name;

        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name = name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_name = name;
            }
        }

        if (input_name.empty() || output_name.empty()) {
            throw std::runtime_error("Failed to find input or output tensor name.");
        }

        std::cout << "TensorRT input name: " << input_name << std::endl;
        std::cout << "TensorRT output name: " << output_name << std::endl;

        int batch_size = static_cast<int>(image_paths.size());

        std::vector<float> input_data(batch_size * 3 * 224 * 224);

        for (int i = 0; i < batch_size; ++i) {
            auto chw = preprocess_image(image_paths[i]);
            std::memcpy(
                input_data.data() + i * 3 * 224 * 224,
                chw.data(),
                3 * 224 * 224 * sizeof(float)
            );
        }

        nvinfer1::Dims input_dims;
        input_dims.nbDims = 4;
        input_dims.d[0] = batch_size;
        input_dims.d[1] = 3;
        input_dims.d[2] = 224;
        input_dims.d[3] = 224;

        if (!context->setInputShape(input_name.c_str(), input_dims)) {
            throw std::runtime_error("Failed to set input shape.");
        }

        auto output_dims = context->getTensorShape(output_name.c_str());

        std::cout << "Output dims: ";
        for (int i = 0; i < output_dims.nbDims; ++i) {
            std::cout << output_dims.d[i] << " ";
        }
        std::cout << std::endl;

        int output_count = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_count *= output_dims.d[i];
        }

        std::vector<float> output_data(output_count);

        float* d_input = nullptr;
        float* d_output = nullptr;
        cudaStream_t stream = nullptr;

        size_t input_bytes = input_data.size() * sizeof(float);
        size_t output_bytes = output_data.size() * sizeof(float);

        check_cuda(cudaMalloc(&d_input, input_bytes), "cudaMalloc d_input failed");
        check_cuda(cudaMalloc(&d_output, output_bytes), "cudaMalloc d_output failed");
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");

        check_cuda(
            cudaMemcpyAsync(d_input, input_data.data(), input_bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D failed"
        );

        if (!context->setTensorAddress(input_name.c_str(), d_input)) {
            throw std::runtime_error("Failed to set input tensor address.");
        }

        if (!context->setTensorAddress(output_name.c_str(), d_output)) {
            throw std::runtime_error("Failed to set output tensor address.");
        }

        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 failed.");
        }

        check_cuda(
            cudaMemcpyAsync(output_data.data(), d_output, output_bytes, cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync D2H failed"
        );

        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed");

        // 简单测速：预热 + 多次重复
        int warmup_times = 5;
        int repeat_times = 50;

        for (int i = 0; i < warmup_times; ++i) {
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("TensorRT warmup enqueueV3 failed.");
            }
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after warmup failed");

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < repeat_times; ++i) {
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("TensorRT timing enqueueV3 failed.");
            }
        }
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after timing failed");

        auto end = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / repeat_times;

        std::cout << "TensorRT warmup times: " << warmup_times << std::endl;
        std::cout << "TensorRT repeat times: " << repeat_times << std::endl;
        std::cout << "TensorRT average inference time: " << avg_ms << " ms" << std::endl;
        std::cout << "TensorRT throughput: " << (batch_size * 1000.0 / avg_ms) << " images/s" << std::endl;

        std::cout << "\nTop-5 prediction results:\n" << std::endl;

        for (int b = 0; b < batch_size; ++b) {
            std::vector<float> logits(
                output_data.begin() + b * 1000,
                output_data.begin() + (b + 1) * 1000
            );

            auto probs = softmax(logits);
            auto top5 = topk_indices(probs, 5);

            std::cout << "Image: " << image_paths[b] << std::endl;

            for (int rank = 0; rank < 5; ++rank) {
                int class_id = top5[rank];
                float score = probs[class_id];
                std::string label = (class_id < static_cast<int>(labels.size())) ? labels[class_id] : "Unknown";

                std::cout << rank + 1
                          << ". class_id=" << class_id
                          << ", score=" << score
                          << ", label=" << label
                          << std::endl;
            }

            std::cout << std::endl;
        }

        if (stream) cudaStreamDestroy(stream);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}