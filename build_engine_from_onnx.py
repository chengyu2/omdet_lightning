import tensorrt as trt
import pycuda.driver as cuda

min_batch_size = 1
opt_batch_size = 16
max_batch_size = 32 # Need to be the same as the ONNX graph batch size. Note: Dynamic batch size doesn't work



def build_engine(onnx_file_path, engine_file_path):
    # 初始化TensorRT的日志记录器
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 创建一个TensorRT构建器
    builder = trt.Builder(TRT_LOGGER)

    # 创建一个TensorRT网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 创建一个ONNX解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 读取ONNX模型文件
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 配置构建器
    config = builder.create_builder_config()
    # 对于TensorRT 10.0及更高版本，使用set_memory_pool_limit替代max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30)  # 10GB
    # Set the precision mode to FP16, assuming hardware supports it
    config.set_flag(trt.BuilderFlag.FP16)
    # 获取输入张量
    input_tensors = [network.get_input(i) for i in range(network.num_inputs)]
    input_names = [tensor.name for tensor in input_tensors]

    # 允许动态批量大小
    profile = builder.create_optimization_profile()
    # 为每个输入张量设置维度信息
    for i, input_tensor in enumerate(input_tensors):
        if input_names[i] == 'img_tensor':
            min_shape = (min_batch_size, 3, 640, 640)
            opt_shape = (opt_batch_size, 3, 640, 640)
            max_shape = (max_batch_size, 3, 640, 640)
        elif input_names[i] == 'label_feats':
            min_shape = (80, min_batch_size, 512)
            opt_shape = (80, opt_batch_size, 512)
            max_shape = (80, max_batch_size, 512)
        elif input_names[i] == 'task_feats':
            min_shape = (77, min_batch_size, 512)
            opt_shape = (77, opt_batch_size, 512)
            max_shape = (77, max_batch_size, 512)
        elif input_names[i] == 'task_mask':
            min_shape = (min_batch_size, 77)
            opt_shape = (opt_batch_size, 77)
            max_shape = (max_batch_size, 77)
        else:
            raise ValueError(f"Unknown input tensor: {input_names[i]}")

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    # 构建TensorRT引擎
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # 保存TensorRT引擎
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Done!")
    return engine

if __name__ == "__main__":
    onnx_file_path = "./omdet.onnx"
    engine_file_path = "./omdet.engine"

    build_engine(onnx_file_path, engine_file_path)