// main_npu_sim.cpp
#include <torch/script.h>
#include <torch/torch.h>
#include <nvml.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <iomanip>

namespace fs = std::filesystem;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;
using Milli   = std::chrono::duration<double, std::milli>;

// NPU “hardware” parameters
constexpr int   NPU_EUs       = 8;
constexpr int   NPU_MEM_BANKS = 16;
constexpr float COMM_LAT_MS   = 0.2f; // ms per image communication

// GPU→die‐area lookup
static const std::unordered_map<std::string,float> kDieAreaMap = {
    {"GA106", 276.0f},
    {"GA104", 392.0f},
    {"GA102", 628.0f},
};

void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
void checkNvml(nvmlReturn_t e, const char *msg) {
    if (e != NVML_SUCCESS) {
        std::cerr << "NVML error (" << msg << "): "
                  << nvmlErrorString(e) << "\n";
        std::exit(1);
    }
}

// Measure idle power
double measure_idle(nvmlDevice_t dev) {
    const int iters = 100;
    double sum = 0;
    for (int i = 0; i < iters; ++i) {
        unsigned p0, p1;
        checkNvml(nvmlDeviceGetPowerUsage(dev, &p0), "idle_p0");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        checkNvml(nvmlDeviceGetPowerUsage(dev, &p1), "idle_p1");
        sum += double(int(p1) - int(p0)) / 1000.0;
    }
    return sum / iters;
}

// Runs NPU sim for RUN_SEC, returns avg latency, avg power, throughput & loops count
void measure_npu(
    torch::jit::script::Module& m,
    const std::vector<torch::Tensor>& imgs,
    nvmlDevice_t nvDev,
    double idle_baseline,
    double RUN_SEC,
    double &out_avg_ms,
    double &out_avg_w,
    double &out_tput,
    int    &out_loops
) {
    const int WARM_UP = 5;
    size_t N = imgs.size();
    // Warm-up
    for (int i = 0; i < WARM_UP; ++i) {
        for (auto &img : imgs) {
            std::this_thread::sleep_for(Milli(COMM_LAT_MS));
            m.forward({img});
        }
        checkCuda(cudaDeviceSynchronize());
    }
    // Timed loops
    auto T0 = Clock::now();
    double sum_time = 0, sum_energy = 0;
    size_t total_images = 0;
    int loops = 0;
    while (Seconds(Clock::now() - T0).count() < RUN_SEC) {
        unsigned p0, p1;
        checkNvml(nvmlDeviceGetPowerUsage(nvDev, &p0), "p0");
        auto t1 = Clock::now();
        for (auto &img : imgs) {
            std::this_thread::sleep_for(Milli(COMM_LAT_MS));
            m.forward({img});
        }
        checkCuda(cudaDeviceSynchronize());
        auto t2 = Clock::now();
        checkNvml(nvmlDeviceGetPowerUsage(nvDev, &p1), "p1");

        double dt_ms     = Milli(t2 - t1).count();
        double batch_pow = double(int(p1) - int(p0)) / 1000.0 - idle_baseline;
        if (batch_pow < 0) batch_pow = 0;

        sum_time   += dt_ms;
        sum_energy += batch_pow * (dt_ms/1000.0);
        total_images += N;
        loops++;
    }
    out_loops     = loops;
    out_avg_ms    = sum_time   / out_loops;       // ms per image
    out_avg_w     = sum_energy / RUN_SEC;            // average W
    out_tput      = total_images / RUN_SEC;          // img/s
}

int main(){
    std::ofstream csv("results_npu.csv");
    csv << "model,avg_latency_ms,avg_delta_power_w,throughput_items_per_s,area_used_mm2\n";

    checkNvml(nvmlInit(), "Init");
    nvmlDevice_t nvDev;
    checkNvml(nvmlDeviceGetHandleByIndex(0, &nvDev), "GetHandle");

    // Die‐area lookup
    char nameBuf[64];
    checkNvml(nvmlDeviceGetName(nvDev, nameBuf, sizeof(nameBuf)), "GetName");
    std::string gpuName(nameBuf);
    float die_area = 0;
    for (auto &kv : kDieAreaMap)
        if (gpuName.find(kv.first) != std::string::npos) die_area = kv.second;
    if (!die_area && gpuName.find("RTX 3060") != std::string::npos) die_area = 276.0f;
    if (!die_area) { std::cerr<<"❌ Unknown GPU\n"; return 1; }

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    int gpu_sms = prop.multiProcessorCount;
    double sm_area   = die_area * 0.7  / gpu_sms;
    double bank_area = die_area * 0.3  / NPU_MEM_BANKS;

    // load real images
    torch::Device device(torch::kCUDA);
    std::vector<torch::Tensor> imgs;
    for (auto &p : fs::directory_iterator("real_images/")) {
        cv::Mat im = cv::imread(p.path().string());
        if (im.empty()) continue;
        cv::resize(im, im, {224,224});
        auto t = torch::from_blob(im.data,{1,224,224,3},torch::kUInt8)
                     .permute({0,3,1,2})
                     .to(torch::kFloat32).div(255)
                     .to(device);
        imgs.push_back(t);
    }
    if (imgs.empty()) { std::cerr<<"❌ No images\n"; return 1; }

    std::vector<std::string> models = {
      "exported/alexnet.pt",      "exported/densenet121.pt",
      "exported/efficientnet_b0.pt","exported/mobilenet_v2.pt",
      "exported/mobilenet_v3_small.pt","exported/regnet_y_400mf.pt",
      "exported/resnet18.pt",     "exported/shufflenet_v2_x1_0.pt",
      "exported/squeezenet1_0.pt", "exported/vgg11.pt"
    };

    constexpr double RUN_SEC = 30.0;
    double idle_baseline = measure_idle(nvDev);

    for (auto &path : models) {
        // strip exported/ and .pt
        std::string name = fs::path(path).stem().string();

        std::cout << "\n=== " << name << " ===\n";
        torch::jit::script::Module m;
        try {
            m = torch::jit::load(path, device);
            m.eval();
        } catch (const std::exception &e) {
            std::cerr<<"⚠️ Load failed: "<<e.what()<<"\n";
            continue;
        }

        double avg_ms, avg_w, tput; 
        int loops;
        measure_npu(m, imgs, nvDev, idle_baseline, RUN_SEC,
                    avg_ms, avg_w, tput, loops);

        nvmlUtilization_t util;
        checkNvml(nvmlDeviceGetUtilizationRates(nvDev, &util), "GetUtil");
        double eu_frac = util.gpu / 100.0;
        double area_used = eu_frac * NPU_EUs * sm_area
                         + double(NPU_MEM_BANKS) * bank_area;

        // **Match GPU printout style exactly**
        std::cout << std::fixed
                  << "Avg latency: "   << std::setprecision(3) << avg_ms  << " ms over " << loops << " loops\n"
                  << "Avg Δ Power: "   << std::setprecision(5) << avg_w   << " W\n"
                  << "Throughput : "   << std::setprecision(4) << tput   << " img/s\n"
                  << "Area used  : "   << std::setprecision(4) << area_used << " mm^2\n";

        csv << name  << ","
            << std::fixed << std::setprecision(3) << avg_ms  << ","
            << std::fixed << std::setprecision(5) << avg_w   << ","
            << std::fixed << std::setprecision(4) << tput   << ","
            << std::fixed << std::setprecision(4) << area_used
            << "\n";
    }

    nvmlShutdown();
    csv.close();
    std::cout<<"\nWrote results_npu.csv\n";
    return 0;
}
