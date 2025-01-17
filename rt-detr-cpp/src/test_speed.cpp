#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <chrono>
#include <numeric>

using torch::indexing::None;
using torch::indexing::Slice;

int main()
{
    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (torch::cuda::is_available() ? "GPU" : "CPU") << std::endl;

    try
    {
        // Load the model
        std::cout << "Loading model..." << std::endl;
        std::string model_path = "/home/appuser/object_detection/rt-detr-cpp/best.torchscript";
        torch::jit::script::Module model;
        model = torch::jit::load(model_path, device);
        model.eval();

        // Create a dummy input tensor (1, 3, 640, 640)
        auto dummy_input = torch::ones({1, 3, 640, 640}, torch::TensorOptions().device(device));

        // Warmup runs
        std::cout << "Warming up..." << std::endl;
        const int warmup_runs = 10;
        for (int i = 0; i < warmup_runs; i++)
        {
            torch::NoGradGuard no_grad;
            model.forward({dummy_input});
            if (torch::cuda::is_available())
            {
                torch::cuda::synchronize();
            }
        }

        // Timing runs
        std::cout << "Running speed test..." << std::endl;
        const int test_runs = 100;
        std::vector<double> times;
        times.reserve(test_runs);

        for (int i = 0; i < test_runs; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();

            torch::NoGradGuard no_grad;
            model.forward({dummy_input});
            if (torch::cuda::is_available())
            {
                torch::cuda::synchronize();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // Convert to milliseconds
        }

        // Calculate statistics
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();

        std::vector<double> diff(times.size());
        std::transform(times.begin(), times.end(), diff.begin(),
                       [mean](double x)
                       { return x - mean; });

        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double std_dev = std::sqrt(sq_sum / times.size());

        // Calculate FPS
        double fps = 1000.0 / mean;

        // Report results
        std::cout << "\nSpeed Test Results:" << std::endl;
        std::cout << "Average time: " << mean << " ms" << std::endl;
        std::cout << "Standard deviation: " << std_dev << " ms" << std::endl;
        std::cout << "FPS: " << fps << std::endl;
    }
    catch (const c10::Error &e)
    {
        std::cout << e.msg() << std::endl;
    }

    return 0;
}