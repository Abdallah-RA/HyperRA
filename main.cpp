#include <torch/script.h>
#include <torch/torch.h>
#include <nvml.h>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;
using Milli   = std::chrono::duration<double, std::milli>;

int main() {
    std::ofstream csv("results.csv");
    csv << "model,avg_latency_ms,avg_delta_power_w,throughput_items_per_s,area_used_mm2\n";

    // 1) Init NVML
    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "❌ NVML Init failed\n";
        return -1;
    }
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex(0, &dev);

    // 2) Die area shares
    const double total_area_mm2 = 276.0;  // GA106
    const double comp_share     = 0.7;
    const double mem_share      = 0.3;

    // 3) Preload & preprocess vision inputs
    torch::Device device(torch::kCUDA);
    std::vector<torch::Tensor> vision_inputs;
    for (auto &p : fs::directory_iterator("real_images/")) {
        cv::Mat img = cv::imread(p.path().string());
        if (img.empty()) continue;
        cv::resize(img, img, {224,224});
        auto t = torch::from_blob(
            img.data, {1,224,224,3}, torch::kUInt8)
            .permute({0,3,1,2})
            .to(torch::kFloat32)
            .div(255)
            .to(device);
        vision_inputs.push_back(t);
    }
    size_t num_images = vision_inputs.size();
    if (num_images == 0) {
        std::cerr << "❌ No images found in real_images/\n";
        return 1;
    }

    // 4) Load pre-tokenized text data as a TorchScript module
    torch::Tensor all_input_ids, all_attention_mask;
    try {
        auto text_data = torch::jit::load("exported/text_data.pt", device);
        text_data.eval();
        auto tup = text_data.forward({}).toTuple();
        all_input_ids      = tup->elements()[0].toTensor();
        all_attention_mask = tup->elements()[1].toTensor();
    } catch (const std::exception &e) {
        std::cerr << "⚠️ Failed loading text_data.pt: " << e.what() << "\n";
    }
    size_t num_texts = all_input_ids.defined() ? all_input_ids.size(0) : 0;

    // 5) All models
    std::vector<std::string> models = {
        "exported/alexnet.pt",
        "exported/densenet121.pt",
        "exported/distilbert-base-uncased.pt",
        "exported/efficientnet_b0.pt",
        "exported/facebook_opt-125m.pt",
        "exported/google_mobilebert-uncased.pt",
        "exported/mobilenet_v2.pt",
        "exported/mobilenet_v3_small.pt",
        "exported/prajjwal1_bert-tiny.pt",
        "exported/regnet_y_400mf.pt",
        "exported/resnet18.pt",
        "exported/sentence-transformers_all-MiniLM-L6-v2.pt",
        "exported/shufflenet_v2_x1_0.pt",
        "exported/squeezenet1_0.pt",
        "exported/sshleifer_tiny-gpt2.pt",
        "exported/vgg11.pt"
    };

    const int WARM_UP     = 5;
    const int MEASURE_SEC = 30;
    const int COOLDOWN    = 5;

    // 6) Loop models
    for (auto &path : models) {
        std::cout << "\n=== " << path << " ===\n";
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(path, device);
            module.eval();
        } catch (const std::exception &e) {
            std::cerr << "⚠️  Load failed: " << e.what() << "\n";
            continue;
        }

        bool is_nlp = (path.find("bert")!=std::string::npos ||
                       path.find("gpt2")!=std::string::npos ||
                       path.find("MiniLM")!=std::string::npos);
        size_t items = is_nlp ? num_texts : num_images;
        if (items == 0) {
            std::cerr << "⚠️  No workload for " << path << "\n";
            continue;
        }

        // Warm-up
        try {
            for (int i = 0; i < WARM_UP; ++i) {
                if (is_nlp) {
                    auto ids  = all_input_ids[0].unsqueeze(0);
                    auto mask = all_attention_mask[0].unsqueeze(0);
                    module.forward({ids, mask});
                } else {
                    module.forward({vision_inputs[0]});
                }
                torch::cuda::synchronize();
            }
        } catch (...) {
            std::cerr << "⚠️  Warm-up error: " << path << "\n";
            continue;
        }

        // Measurement
        auto t_start = Clock::now();
        double sum_ms = 0.0, sum_w = 0.0;
        double sum_comp = 0.0, sum_mem = 0.0;
        int runs = 0;

        while (Seconds(Clock::now() - t_start).count() < MEASURE_SEC) {
            nvmlUtilization_t util;     nvmlDeviceGetUtilizationRates(dev, &util);
            nvmlMemory_t meminfo;       nvmlDeviceGetMemoryInfo(dev, &meminfo);
            sum_comp += util.gpu / 100.0;
            sum_mem  += double(meminfo.used) / double(meminfo.total);

            unsigned int p0=0, p1=0;
            nvmlDeviceGetPowerUsage(dev, &p0);
            auto t0 = Clock::now();

            if (is_nlp) {
                for (size_t i = 0; i < num_texts; ++i) {
                    auto ids  = all_input_ids[i].unsqueeze(0);
                    auto mask = all_attention_mask[i].unsqueeze(0);
                    module.forward({ids, mask});
                }
            } else {
                for (auto &inp : vision_inputs) {
                    module.forward({inp});
                }
            }
            torch::cuda::synchronize();

            auto t1 = Clock::now();
            nvmlDeviceGetPowerUsage(dev, &p1);

            sum_ms += Milli(t1 - t0).count();
            sum_w  += double(int(p1) - int(p0)) / 1000.0;
            runs++;
        }

        if (runs == 0) {
            std::cerr << "⚠️  No runs: " << path << "\n";
            continue;
        }

        double avg_ms     = sum_ms / runs;
        double avg_w      = sum_w  / runs;
        double throughput = double(runs) * double(items) / MEASURE_SEC;
        double avg_cu     = sum_comp / runs;
        double avg_mu     = sum_mem  / runs;
        double area_used  = total_area_mm2 * (comp_share * avg_cu + mem_share * avg_mu);
        if (avg_w < 0 && avg_w > -0.5) avg_w = 0.0;

        std::cout << "Avg latency: " << avg_ms   << " ms over " << runs << " loops\n";
        std::cout << "Avg Δ Power: "  << avg_w    << " W\n";
        std::cout << "Throughput : "  << throughput << " items/s\n";
        std::cout << "Area used  : "  << area_used << " mm^2\n";

        csv << path
            << "," << avg_ms
            << "," << avg_w
            << "," << throughput
            << "," << area_used << "\n";
        csv.flush();

        std::this_thread::sleep_for(std::chrono::seconds(COOLDOWN));
    }

    nvmlShutdown();
    csv.close();
    std::cout << "\nResults written to results.csv\n";
    return 0;
}
