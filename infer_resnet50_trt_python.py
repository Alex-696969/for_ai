import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def check_cuda_error(err, msg="CUDA error"):
    # cuda-python / cuda-bindings 有些 API 返回的是：
    # (cudaError_t.cudaSuccess,)
    # 所以这里先把 tuple 里的第一个元素取出来
    if isinstance(err, tuple):
        err = err[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{msg}: {err}")


def load_labels(labels_path):
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


def resize_short_side(image_rgb, short_side=232):
    h, w = image_rgb.shape[:2]

    if h < w:
        new_h = short_side
        new_w = int(round(w * short_side / h))
    else:
        new_w = short_side
        new_h = int(round(h * short_side / w))

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def center_crop(image_rgb, crop_size=224):
    h, w = image_rgb.shape[:2]

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    cropped = image_rgb[top:top + crop_size, left:left + crop_size]

    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        raise RuntimeError(
            f"Center crop failed. Expected {crop_size}x{crop_size}, "
            f"but got {cropped.shape[0]}x{cropped.shape[1]}"
        )

    return cropped


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # OpenCV 默认读取的是 BGR，而 torchvision ResNet-50 预训练模型使用 RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 与 torchvision ResNet-50 官方预处理保持一致：
    # resize 短边到 232，再 center crop 到 224x224
    image_rgb = resize_short_side(image_rgb, short_side=232)
    image_rgb = center_crop(image_rgb, crop_size=224)

    # 转 float32，并归一化到 [0, 1]
    image = image_rgb.astype(np.float32) / 255.0

    # ImageNet 标准化参数，顺序是 RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))

    # TensorRT 输入要求连续内存
    image = np.ascontiguousarray(image, dtype=np.float32)

    return image


def softmax(logits):
    logits = logits.astype(np.float32)

    # 减去最大值，避免 exp 溢出
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp = np.exp(logits)
    prob = exp / np.sum(exp, axis=1, keepdims=True)

    return prob


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    logger = trt.Logger(trt.Logger.INFO)

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")

    return engine


def get_input_output_names(engine):
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    if len(input_names) != 1:
        raise RuntimeError(
            f"Expected 1 input tensor, but got {len(input_names)}: {input_names}"
        )

    if len(output_names) != 1:
        raise RuntimeError(
            f"Expected 1 output tensor, but got {len(output_names)}: {output_names}"
        )

    return input_names[0], output_names[0]


def tensor_dtype_to_numpy(dtype):
    return trt.nptype(dtype)


def allocate_device_memory(nbytes, name):
    err, device_ptr = cudart.cudaMalloc(nbytes)
    check_cuda_error(err, f"cudaMalloc failed for {name}")

    return device_ptr


def free_device_memory(device_ptr, name):
    if device_ptr is not None:
        err = cudart.cudaFree(device_ptr)
        check_cuda_error(err, f"cudaFree failed for {name}")


def run_inference(engine_path, image_paths, labels_path, warmup_times=5, repeat_times=50):
    labels = load_labels(labels_path)

    images = []

    for image_path in image_paths:
        image = preprocess_image(image_path)
        images.append(image)

    input_array = np.stack(images, axis=0)
    input_array = np.ascontiguousarray(input_array, dtype=np.float32)

    batch_size = input_array.shape[0]

    print(f"Input batch shape: {input_array.shape}")

    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_name, output_name = get_input_output_names(engine)

    print(f"TensorRT input name: {input_name}")
    print(f"TensorRT output name: {output_name}")

    # 设置动态 batch 的实际输入 shape
    # 例如 batch=1 时是 (1, 3, 224, 224)
    # 例如 batch=3 时是 (3, 3, 224, 224)
    ok = context.set_input_shape(input_name, input_array.shape)

    if not ok:
        raise RuntimeError(f"Failed to set input shape: {input_array.shape}")

    output_shape = tuple(context.get_tensor_shape(output_name))
    output_dtype = tensor_dtype_to_numpy(engine.get_tensor_dtype(output_name))

    print(f"TensorRT output shape: {output_shape}")
    print(f"TensorRT output dtype: {output_dtype}")

    output_array = np.empty(output_shape, dtype=output_dtype)

    input_nbytes = input_array.nbytes
    output_nbytes = output_array.nbytes

    print(f"Input bytes: {input_nbytes}")
    print(f"Output bytes: {output_nbytes}")

    input_device = None
    output_device = None
    stream = None

    try:
        input_device = allocate_device_memory(input_nbytes, "input")
        output_device = allocate_device_memory(output_nbytes, "output")

        err, stream = cudart.cudaStreamCreate()
        check_cuda_error(err, "cudaStreamCreate failed")

        # 将输入从 CPU 内存拷贝到 GPU 显存
        err = cudart.cudaMemcpyAsync(
            input_device,
            input_array.ctypes.data,
            input_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
        check_cuda_error(err, "cudaMemcpyAsync HostToDevice failed")

        # 绑定 TensorRT 输入输出 Tensor 到 GPU 显存地址
        context.set_tensor_address(input_name, int(input_device))
        context.set_tensor_address(output_name, int(output_device))

        # 第一次真实推理，用于得到输出结果
        ok = context.execute_async_v3(stream_handle=stream)

        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        err = cudart.cudaStreamSynchronize(stream)
        check_cuda_error(err, "cudaStreamSynchronize after first inference failed")

        # 预热，避免第一次推理的初始化开销影响计时
        for _ in range(warmup_times):
            ok = context.execute_async_v3(stream_handle=stream)

            if not ok:
                raise RuntimeError("TensorRT warmup execute_async_v3 failed.")

        err = cudart.cudaStreamSynchronize(stream)
        check_cuda_error(err, "cudaStreamSynchronize after warmup failed")

        # 正式计时
        start_time = time.perf_counter()

        for _ in range(repeat_times):
            ok = context.execute_async_v3(stream_handle=stream)

            if not ok:
                raise RuntimeError("TensorRT timing execute_async_v3 failed.")

        err = cudart.cudaStreamSynchronize(stream)
        check_cuda_error(err, "cudaStreamSynchronize after timing failed")

        end_time = time.perf_counter()

        avg_ms = (end_time - start_time) * 1000.0 / repeat_times

        print()
        print(f"TensorRT warmup times: {warmup_times}")
        print(f"TensorRT repeat times: {repeat_times}")
        print(f"TensorRT average inference time: {avg_ms:.3f} ms")
        print(f"TensorRT throughput: {batch_size * 1000.0 / avg_ms:.2f} images/s")

        # 将输出从 GPU 显存拷贝回 CPU 内存
        err = cudart.cudaMemcpyAsync(
            output_array.ctypes.data,
            output_device,
            output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        check_cuda_error(err, "cudaMemcpyAsync DeviceToHost failed")

        err = cudart.cudaStreamSynchronize(stream)
        check_cuda_error(err, "cudaStreamSynchronize after output copy failed")

    finally:
        if stream is not None:
            err = cudart.cudaStreamDestroy(stream)
            check_cuda_error(err, "cudaStreamDestroy failed")

        free_device_memory(input_device, "input")
        free_device_memory(output_device, "output")

    probabilities = softmax(output_array)

    print()
    print("Top-5 prediction results:")

    for batch_index, image_path in enumerate(image_paths):
        prob = probabilities[batch_index]

        top5_indices = np.argsort(prob)[-5:][::-1]

        print()
        print(f"Image: {image_path}")

        for rank, class_id in enumerate(top5_indices, start=1):
            score = prob[class_id]

            if class_id < len(labels):
                class_name = labels[class_id]
            else:
                class_name = "Unknown"

            print(
                f"{rank}. class_id={class_id}, "
                f"score={score:.6f}, "
                f"label={class_name}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        type=str,
        default="engines/resnet50_fp32.plan",
        help="Path to TensorRT engine file.",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="exports/imagenet_classes.txt",
        help="Path to ImageNet labels file.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup inference times.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=50,
        help="Repeated inference times for performance measurement.",
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="Input image paths.",
    )

    args = parser.parse_args()

    run_inference(
        engine_path=args.engine,
        image_paths=args.images,
        labels_path=args.labels,
        warmup_times=args.warmup,
        repeat_times=args.repeat,
    )


if __name__ == "__main__":
    main()