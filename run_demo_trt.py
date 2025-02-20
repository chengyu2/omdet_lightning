import os
import time
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import concurrent.futures


def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        # 获取绑定的形状信息
        binding_shape = engine.get_tensor_shape(binding)
        if binding == 0:  # 假设第一个绑定是输入，并且批量维度是第一个维度
            binding_shape[0] = batch_size
        # 计算缓冲区大小
        size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # 分配主机和设备缓冲区
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 将设备缓冲区添加到绑定列表中
        bindings.append(int(device_mem))
        # 根据绑定类型添加到相应的列表中
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # 将输入数据传输到GPU
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    # 设置绑定信息
    engine = context.engine
    for i, binding in enumerate(bindings):
        binding_name = engine.get_tensor_name(i)
        context.set_tensor_address(binding_name, binding)
    # 运行推理
    context.execute_async_v3(stream_handle=stream.handle)
    # 将预测结果从GPU传输回主机
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    # 同步流
    stream.synchronize()
    # 只返回主机输出
    return [out[0] for out in outputs]


def letterbox_image(image, size):
    """
    Resize the image while maintaining the aspect ratio and pad the remaining area.
    :param image: PIL Image object
    :param size: Target size, e.g., (640, 640)
    :return: Resized and padded PIL Image object
    """
    ih, iw = image.height, image.width
    h, w = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess_image(image_path):
    """
    Preprocess the image, including resizing, padding, and normalization.
    :param image_path: Path to the input image
    :return: Numpy array of shape [1, 3, 640, 640]
    """
    # Open the image
    image = Image.open(image_path)

    # Resize and pad the image
    resized_image = letterbox_image(image, (640, 640))

    # Convert the image to a numpy array
    image_array = np.array(resized_image, dtype=np.float32)

    # Normalize the pixel values to [0, 1]
    image_array /= 255.0

    # Transpose the array to [3, 640, 640]
    image_array = np.transpose(image_array, (2, 0, 1))

    # Add a batch dimension to get [1, 3, 640, 640]
    image_array = np.expand_dims(image_array, axis=0)

    return image_array



def preprocess_batch(image_paths):
    """
    Preprocess a batch of images.
    :param image_paths: List of paths to input images
    :return: Numpy array of shape [batch_size, 3, 640, 640]
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit preprocess_image tasks for each image path
        futures = [executor.submit(preprocess_image, img_path) for img_path in image_paths]
        # Gather the results
        batch_images = [future.result() for future in concurrent.futures.as_completed(futures)]
    batch_images = np.concatenate(batch_images, axis=0)
    return batch_images


if __name__ == "__main__":
    engine_path = './omdet.engine'
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    img_paths = ['./sample_data/000000574769.jpg', './sample_data/161030_nns_machette_16x9_992.jpg'] * 1000  # 2000 images total
    batch_size = 16  # 可根据实际情况调整批量大小
    labels = ["person", "cat", "orange"]  # 要预测的标签
    prompt = 'Detect {}.'.format(','.join(labels))  # 检测任务的提示，使用 "Detect {}." 作为默认

    start_time = time.time()

    for i in range(0, len(img_paths), batch_size):
        batch_img_paths = img_paths[i:i + batch_size]
        # 预处理批量图像
        batch_img_array = preprocess_batch(batch_img_paths)

        inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size)

        # 将输入数据复制到输入缓冲区
        host_mem = inputs[0][0]
        np.copyto(host_mem, batch_img_array.ravel())

        # 运行推理
        trt_outputs = do_inference(context, bindings, inputs, outputs, stream)

        # 后处理输出
        # 这里需要将TensorRT的输出转换为与原始PyTorch模型输出相同的格式
        # 然后执行与run_demo.py中相同的后处理步骤
        # 为了简单起见，假设输出是一个边界框和分数的列表
        # Infernece result is a list of 2
        # Item 0: [0.17500001 0.52500004 0.2        ... 0.31875    0.05000001 0.05000001] Shape (3600, 0)
        # Item 1: [0.3485693  0.3485693  0.3485693  ... 0.70701766 0.70701766 0.70701766] Shape (72000, 0)
        # TODO how to interpret the result

        out_folder = './outputs'
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

        # # 保存带有注释的图像（你需要实现注释部分）
        # for j, img_path in enumerate(batch_img_paths):
        #     im = Image.open(img_path)
        #     im.save(os.path.join(out_folder, img_path.split('/')[-1]))

    total_time = time.time() - start_time
    throughput = len(img_paths) / total_time
    print(f"Inference took {total_time:.2f} seconds for {len(img_paths)} images. Throughput: {throughput:.2f} images/second.")
