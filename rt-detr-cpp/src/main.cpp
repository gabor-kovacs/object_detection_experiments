﻿#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using torch::indexing::None;
using torch::indexing::Slice;

float generate_scale(cv::Mat &image, const std::vector<int> &target_size)
{
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size)
{
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0])
    {
        if (input_image.data == output_image.data)
        {
            return 1.;
        }
        else
        {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}

torch::Tensor xyxy2xywh(const torch::Tensor &x)
{
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}

torch::Tensor xywh2xyxy(const torch::Tensor &x)
{
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}

// Reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor nms(const torch::Tensor &bboxes, const torch::Tensor &scores, float iou_threshold)
{
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++)
    {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++)
        {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor non_max_suppression(torch::Tensor &prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300)
{
    auto bs = prediction.size(0);
    auto nc = prediction.size(2) - 4; // number of classes (7 for your case)

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++)
    {
        output.push_back(torch::zeros({0, 6}, prediction.options())); // 6 = 4 box + conf + cls
    }

    for (int xi = 0; xi < prediction.size(0); xi++)
    {
        auto x = prediction[xi];

        // Get boxes and scores
        auto boxes = x.slice(1, 0, 4);
        auto scores = x.slice(1, 4, 11); // class scores start at index 4

        // Convert boxes from [x,y,w,h] to [x1,y1,x2,y2]
        boxes = xywh2xyxy(boxes);

        // Get max confidence and class
        auto [conf, j] = scores.max(1);

        // Filter by confidence
        auto conf_mask = conf > conf_thres;
        boxes = boxes.index({conf_mask});
        conf = conf.index({conf_mask});
        j = j.index({conf_mask});

        if (boxes.size(0) == 0)
            continue;

        // NMS
        auto i = nms(boxes, conf, iou_thres);
        if (i.size(0) > max_det)
            i = i.slice(0, 0, max_det);

        // Combine results
        auto det = torch::cat({boxes.index({i}),
                               conf.index({i}).unsqueeze(1),
                               j.index({i}).toType(torch::kFloat32).unsqueeze(1)},
                              1);
        output[xi] = det;
    }

    return torch::stack(output);
}

torch::Tensor clip_boxes(torch::Tensor &boxes, const std::vector<int> &shape)
{
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}).clamp(0, shape[1]));
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}).clamp(0, shape[0]));
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}).clamp(0, shape[1]));
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}).clamp(0, shape[0]));
    return boxes;
}

torch::Tensor scale_boxes(const std::vector<int> &img1_shape, torch::Tensor &boxes, const std::vector<int> &img0_shape)
{
    // First scale normalized coords (0-1) to img1 dimensions
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) * img1_shape[1]); // x1 * width
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) * img1_shape[0]); // y1 * height
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) * img1_shape[1]); // x2 * width
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) * img1_shape[0]); // y2 * height

    // Then adjust for padding and scale
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}

int main()
{
    // Device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (torch::cuda::is_available() ? "GPU" : "CPU") << std::endl;

    // Note that in this example the classes are hard-coded
    // std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    //                                  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    //                                  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    //                                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    //                                  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    //                                  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    //                                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    std::vector<std::string> classes{"fish", "jellyfish", "penguin", "puffin", "shark", "starfish", "stingray"};

    try
    {
        std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
        std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;

        // Load the model
        std::cout << "Loading model..." << std::endl;
        std::string model_path = "/home/appuser/object_detection/rt-detr-cpp/best.torchscript";
        torch::jit::script::Module model;
        model = torch::jit::load(model_path, device);
        model.eval();

        // Load image and preprocess
        std::cout << "Loading image..." << std::endl;
        cv::Mat image = cv::imread("/home/appuser/object_detection/images/testim.jpg");
        if (image.empty())
        {
            std::cerr << "Error: Could not read the image." << std::endl;
            return -1;
        }

        cv::Mat input_image;
        letterbox(image, input_image, {640, 640});
        cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

        std::cout << "Creating tensor..." << std::endl;
        torch::Tensor image_tensor = torch::from_blob(input_image.data,
                                                      {input_image.rows, input_image.cols, 3},
                                                      torch::kByte);
        std::cout << "Initial tensor device: " << image_tensor.device() << std::endl;

        image_tensor = image_tensor.to(device);
        std::cout << "After moving to device: " << image_tensor.device() << std::endl;

        image_tensor = image_tensor.toType(torch::kFloat32).div(255);
        image_tensor = image_tensor.permute({2, 0, 1});
        image_tensor = image_tensor.unsqueeze(0);
        std::cout << "Final input tensor device: " << image_tensor.device() << std::endl;

        // Inference
        std::cout << "Running inference..." << std::endl;
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs{image_tensor};

        std::cout << "Created inputs vector" << std::endl;
        auto output = model.forward(inputs);
        std::cout << "Forward pass complete" << std::endl;

        auto output_tensor = output.toTensor();
        std::cout << "Output tensor shape: " << output_tensor.sizes() << std::endl;
        std::cout << "Output tensor first few values: " << output_tensor.slice(1, 0, 10) << std::endl;

        output_tensor = output_tensor.cpu();
        std::cout << "Moved to CPU" << std::endl;

        // NMS
        auto keep = non_max_suppression(output_tensor)[0];
        std::cout << "NMS complete" << std::endl;
        std::cout << "Keep tensor shape: " << keep.sizes() << std::endl;

        std::cout << "Number of detections: " << keep.size(0) << std::endl;

        auto boxes = keep.index({Slice(), Slice(None, 4)});
        std::cout << "Got boxes" << std::endl;

        keep.index_put_({Slice(), Slice(None, 4)},
                        scale_boxes({input_image.rows, input_image.cols}, boxes, {image.rows, image.cols}));
        std::cout << "Scaled boxes" << std::endl;

        // Show the results
        for (int i = 0; i < keep.size(0); i++)
        {
            std::cout << "Processing detection " << i << std::endl;

            int x1 = keep[i][0].item().toFloat();
            int y1 = keep[i][1].item().toFloat();
            int x2 = keep[i][2].item().toFloat();
            int y2 = keep[i][3].item().toFloat();
            float conf = keep[i][4].item().toFloat();
            int cls = keep[i][5].item().toInt();

            std::cout << "Got coordinates for detection " << i << std::endl;
            std::cout << "Class index: " << cls << ", Classes size: " << classes.size() << std::endl;

            std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2
                      << "]  Conf: " << conf << "  Class: " << classes[cls] << std::endl;
        }

        // Draw detections on a copy of the image
        cv::Mat output_image = image.clone();

        for (int i = 0; i < keep.size(0); i++)
        {
            int x1 = keep[i][0].item().toFloat();
            int y1 = keep[i][1].item().toFloat();
            int x2 = keep[i][2].item().toFloat();
            int y2 = keep[i][3].item().toFloat();
            float conf = keep[i][4].item().toFloat();
            int cls = keep[i][5].item().toInt();

            // Draw rectangle
            cv::rectangle(output_image,
                          cv::Point(x1, y1),
                          cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);

            // Prepare label text
            std::string label = classes[cls] + " " + std::to_string(conf).substr(0, 4);

            // Get text size for background rectangle
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                 0.5, 1, &baseline);

            // Draw background rectangle for text
            cv::rectangle(output_image,
                          cv::Point(x1, y1 - text_size.height - 5),
                          cv::Point(x1 + text_size.width, y1),
                          cv::Scalar(0, 255, 0), -1);

            // Draw text
            cv::putText(output_image, label,
                        cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0), 1);
        }

        // Save the result
        cv::imwrite("result.jpg", output_image);
    }
    catch (const c10::Error &e)
    {
        std::cout << e.msg() << std::endl;
    }

    return 0;
}