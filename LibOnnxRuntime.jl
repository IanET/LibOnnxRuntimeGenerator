module LibOnnxRuntime

using CEnum: CEnum, @cenum

using Pkg.Artifacts

@static if Sys.iswindows()
    @static if Sys.ARCH == :x86_64
        const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes\\win-x64\\native\\onnxruntime.dll")
    else # Sys.ARCH == :aarch64
        const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes\\win-arm64\\native\\onnxruntime.dll")
    end
elseif Sys.islinux()
    @static if Sys.ARCH == :x86_64
        const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes/linux-x64/native/libonnxruntime.so")
    else # Sys.ARCH == :aarch64
        const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes/linux-arm64/native/libonnxruntime.so")
    end
end

mutable struct OrtStatus end

const OrtStatusPtr = Ptr{OrtStatus}

@cenum ONNXTensorElementDataType::UInt32 begin
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN = 17
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ = 18
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 = 19
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = 20
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 = 21
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 = 22
end

@cenum ONNXType::UInt32 begin
    ONNX_TYPE_UNKNOWN = 0
    ONNX_TYPE_TENSOR = 1
    ONNX_TYPE_SEQUENCE = 2
    ONNX_TYPE_MAP = 3
    ONNX_TYPE_OPAQUE = 4
    ONNX_TYPE_SPARSETENSOR = 5
    ONNX_TYPE_OPTIONAL = 6
end

@cenum OrtSparseFormat::UInt32 begin
    ORT_SPARSE_UNDEFINED = 0
    ORT_SPARSE_COO = 1
    ORT_SPARSE_CSRC = 2
    ORT_SPARSE_BLOCK_SPARSE = 4
end

@cenum OrtSparseIndicesFormat::UInt32 begin
    ORT_SPARSE_COO_INDICES = 0
    ORT_SPARSE_CSR_INNER_INDICES = 1
    ORT_SPARSE_CSR_OUTER_INDICES = 2
    ORT_SPARSE_BLOCK_SPARSE_INDICES = 3
end

@cenum OrtLoggingLevel::UInt32 begin
    ORT_LOGGING_LEVEL_VERBOSE = 0
    ORT_LOGGING_LEVEL_INFO = 1
    ORT_LOGGING_LEVEL_WARNING = 2
    ORT_LOGGING_LEVEL_ERROR = 3
    ORT_LOGGING_LEVEL_FATAL = 4
end

@cenum OrtErrorCode::UInt32 begin
    ORT_OK = 0
    ORT_FAIL = 1
    ORT_INVALID_ARGUMENT = 2
    ORT_NO_SUCHFILE = 3
    ORT_NO_MODEL = 4
    ORT_ENGINE_ERROR = 5
    ORT_RUNTIME_EXCEPTION = 6
    ORT_INVALID_PROTOBUF = 7
    ORT_MODEL_LOADED = 8
    ORT_NOT_IMPLEMENTED = 9
    ORT_INVALID_GRAPH = 10
    ORT_EP_FAIL = 11
    ORT_MODEL_LOAD_CANCELED = 12
    ORT_MODEL_REQUIRES_COMPILATION = 13
    ORT_NOT_FOUND = 14
end

@cenum OrtOpAttrType::UInt32 begin
    ORT_OP_ATTR_UNDEFINED = 0
    ORT_OP_ATTR_INT = 1
    ORT_OP_ATTR_INTS = 2
    ORT_OP_ATTR_FLOAT = 3
    ORT_OP_ATTR_FLOATS = 4
    ORT_OP_ATTR_STRING = 5
    ORT_OP_ATTR_STRINGS = 6
    ORT_OP_ATTR_GRAPH = 7
    ORT_OP_ATTR_TENSOR = 8
end

mutable struct OrtEnv end

mutable struct OrtMemoryInfo end

mutable struct OrtIoBinding end

mutable struct OrtSession end

mutable struct OrtValue end

mutable struct OrtRunOptions end

mutable struct OrtTypeInfo end

mutable struct OrtTensorTypeAndShapeInfo end

mutable struct OrtMapTypeInfo end

mutable struct OrtSequenceTypeInfo end

mutable struct OrtOptionalTypeInfo end

mutable struct OrtSessionOptions end

mutable struct OrtCustomOpDomain end

mutable struct OrtModelMetadata end

mutable struct OrtThreadPoolParams end

mutable struct OrtThreadingOptions end

mutable struct OrtArenaCfg end

mutable struct OrtPrepackedWeightsContainer end

mutable struct OrtTensorRTProviderOptionsV2 end

mutable struct OrtNvTensorRtRtxProviderOptions end

mutable struct OrtCUDAProviderOptionsV2 end

mutable struct OrtCANNProviderOptions end

mutable struct OrtDnnlProviderOptions end

mutable struct OrtOp end

mutable struct OrtOpAttr end

mutable struct OrtLogger end

mutable struct OrtShapeInferContext end

mutable struct OrtLoraAdapter end

mutable struct OrtValueInfo end

mutable struct OrtNode end

mutable struct OrtGraph end

mutable struct OrtModel end

mutable struct OrtModelCompilationOptions end

mutable struct OrtHardwareDevice end

mutable struct OrtEpDevice end

mutable struct OrtKeyValuePairs end

mutable struct OrtSyncStream end

mutable struct OrtExternalInitializerInfo end

struct OrtAllocator
    version::UInt32
    Alloc::Ptr{Cvoid}
    Free::Ptr{Cvoid}
    Info::Ptr{Cvoid}
    Reserve::Ptr{Cvoid}
    GetStats::Ptr{Cvoid}
    AllocOnStream::Ptr{Cvoid}
end

# typedef void ( ORT_API_CALL * OrtLoggingFunction
const OrtLoggingFunction = Ptr{Cvoid}

@cenum GraphOptimizationLevel::UInt32 begin
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_LAYOUT = 3
    ORT_ENABLE_ALL = 99
end

@cenum ExecutionMode::UInt32 begin
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1
end

@cenum OrtLanguageProjection::UInt32 begin
    ORT_PROJECTION_C = 0
    ORT_PROJECTION_CPLUSPLUS = 1
    ORT_PROJECTION_CSHARP = 2
    ORT_PROJECTION_PYTHON = 3
    ORT_PROJECTION_JAVA = 4
    ORT_PROJECTION_WINML = 5
    ORT_PROJECTION_NODEJS = 6
end

mutable struct OrtKernelInfo end

mutable struct OrtKernelContext end

struct OrtCustomOp
    version::UInt32
    CreateKernel::Ptr{Cvoid}
    GetName::Ptr{Cvoid}
    GetExecutionProviderType::Ptr{Cvoid}
    GetInputType::Ptr{Cvoid}
    GetInputTypeCount::Ptr{Cvoid}
    GetOutputType::Ptr{Cvoid}
    GetOutputTypeCount::Ptr{Cvoid}
    KernelCompute::Ptr{Cvoid}
    KernelDestroy::Ptr{Cvoid}
    GetInputCharacteristic::Ptr{Cvoid}
    GetOutputCharacteristic::Ptr{Cvoid}
    GetInputMemoryType::Ptr{Cvoid}
    GetVariadicInputMinArity::Ptr{Cvoid}
    GetVariadicInputHomogeneity::Ptr{Cvoid}
    GetVariadicOutputMinArity::Ptr{Cvoid}
    GetVariadicOutputHomogeneity::Ptr{Cvoid}
    CreateKernelV2::Ptr{Cvoid}
    KernelComputeV2::Ptr{Cvoid}
    InferOutputShapeFn::Ptr{Cvoid}
    GetStartVersion::Ptr{Cvoid}
    GetEndVersion::Ptr{Cvoid}
    GetMayInplace::Ptr{Cvoid}
    ReleaseMayInplace::Ptr{Cvoid}
    GetAliasMap::Ptr{Cvoid}
    ReleaseAliasMap::Ptr{Cvoid}
end

@cenum OrtAllocatorType::Int32 begin
    OrtInvalidAllocator = -1
    OrtDeviceAllocator = 0
    OrtArenaAllocator = 1
    OrtReadOnlyAllocator = 2
end

@cenum OrtMemType::Int32 begin
    OrtMemTypeCPUInput = -2
    OrtMemTypeCPUOutput = -1
    OrtMemTypeCPU = -1
    OrtMemTypeDefault = 0
end

@cenum OrtDeviceMemoryType::UInt32 begin
    OrtDeviceMemoryType_DEFAULT = 0
    OrtDeviceMemoryType_HOST_ACCESSIBLE = 5
end

@cenum OrtMemoryInfoDeviceType::UInt32 begin
    OrtMemoryInfoDeviceType_CPU = 0
    OrtMemoryInfoDeviceType_GPU = 1
    OrtMemoryInfoDeviceType_FPGA = 2
    OrtMemoryInfoDeviceType_NPU = 3
end

@cenum OrtHardwareDeviceType::UInt32 begin
    OrtHardwareDeviceType_CPU = 0
    OrtHardwareDeviceType_GPU = 1
    OrtHardwareDeviceType_NPU = 2
end

@cenum OrtExecutionProviderDevicePolicy::UInt32 begin
    OrtExecutionProviderDevicePolicy_DEFAULT = 0
    OrtExecutionProviderDevicePolicy_PREFER_CPU = 1
    OrtExecutionProviderDevicePolicy_PREFER_NPU = 2
    OrtExecutionProviderDevicePolicy_PREFER_GPU = 3
    OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE = 4
    OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY = 5
    OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER = 6
end

# typedef OrtStatus * ( ORT_API_CALL * EpSelectionDelegate
const EpSelectionDelegate = Ptr{Cvoid}

# typedef OrtStatus * ( ORT_API_CALL * OrtWriteBufferFunc
const OrtWriteBufferFunc = Ptr{Cvoid}

# typedef OrtStatus * ( ORT_API_CALL * OrtGetInitializerLocationFunc
const OrtGetInitializerLocationFunc = Ptr{Cvoid}

@cenum OrtCudnnConvAlgoSearch::UInt32 begin
    OrtCudnnConvAlgoSearchExhaustive = 0
    OrtCudnnConvAlgoSearchHeuristic = 1
    OrtCudnnConvAlgoSearchDefault = 2
end

struct OrtCUDAProviderOptions
    device_id::Cint
    cudnn_conv_algo_search::OrtCudnnConvAlgoSearch
    gpu_mem_limit::Csize_t
    arena_extend_strategy::Cint
    do_copy_in_default_stream::Cint
    has_user_compute_stream::Cint
    user_compute_stream::Ptr{Cvoid}
    default_memory_arena_cfg::Ptr{OrtArenaCfg}
    tunable_op_enable::Cint
    tunable_op_tuning_enable::Cint
    tunable_op_max_tuning_duration_ms::Cint
end

struct OrtROCMProviderOptions
    device_id::Cint
    miopen_conv_exhaustive_search::Cint
    gpu_mem_limit::Csize_t
    arena_extend_strategy::Cint
    do_copy_in_default_stream::Cint
    has_user_compute_stream::Cint
    user_compute_stream::Ptr{Cvoid}
    default_memory_arena_cfg::Ptr{OrtArenaCfg}
    enable_hip_graph::Cint
    tunable_op_enable::Cint
    tunable_op_tuning_enable::Cint
    tunable_op_max_tuning_duration_ms::Cint
end

struct OrtTensorRTProviderOptions
    device_id::Cint
    has_user_compute_stream::Cint
    user_compute_stream::Ptr{Cvoid}
    trt_max_partition_iterations::Cint
    trt_min_subgraph_size::Cint
    trt_max_workspace_size::Csize_t
    trt_fp16_enable::Cint
    trt_int8_enable::Cint
    trt_int8_calibration_table_name::Ptr{Cchar}
    trt_int8_use_native_calibration_table::Cint
    trt_dla_enable::Cint
    trt_dla_core::Cint
    trt_dump_subgraphs::Cint
    trt_engine_cache_enable::Cint
    trt_engine_cache_path::Ptr{Cchar}
    trt_engine_decryption_enable::Cint
    trt_engine_decryption_lib_path::Ptr{Cchar}
    trt_force_sequential_engine_build::Cint
end

struct OrtMIGraphXProviderOptions
    device_id::Cint
    migraphx_fp16_enable::Cint
    migraphx_fp8_enable::Cint
    migraphx_int8_enable::Cint
    migraphx_use_native_calibration_table::Cint
    migraphx_int8_calibration_table_name::Ptr{Cchar}
    migraphx_save_compiled_model::Cint
    migraphx_save_model_path::Ptr{Cchar}
    migraphx_load_compiled_model::Cint
    migraphx_load_model_path::Ptr{Cchar}
    migraphx_exhaustive_tune::Bool
    migraphx_mem_limit::Csize_t
    migraphx_arena_extend_strategy::Cint
end

struct OrtOpenVINOProviderOptions
    device_type::Ptr{Cchar}
    enable_npu_fast_compile::Cuchar
    device_id::Ptr{Cchar}
    num_of_threads::Csize_t
    cache_dir::Ptr{Cchar}
    context::Ptr{Cvoid}
    enable_opencl_throttling::Cuchar
    enable_dynamic_shapes::Cuchar
end

struct OrtApi
    data::NTuple{3120, UInt8}
end

function Base.getproperty(x::Ptr{OrtApi}, f::Symbol)
    f === :CreateStatus && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :GetErrorCode && return Ptr{Ptr{Cvoid}}(x + 8)
    f === :GetErrorMessage && return Ptr{Ptr{Cvoid}}(x + 16)
    f === :CreateEnv && return Ptr{Ptr{Cvoid}}(x + 24)
    f === :CreateEnvWithCustomLogger && return Ptr{Ptr{Cvoid}}(x + 32)
    f === :EnableTelemetryEvents && return Ptr{Ptr{Cvoid}}(x + 40)
    f === :DisableTelemetryEvents && return Ptr{Ptr{Cvoid}}(x + 48)
    f === :CreateSession && return Ptr{Ptr{Cvoid}}(x + 56)
    f === :CreateSessionFromArray && return Ptr{Ptr{Cvoid}}(x + 64)
    f === :Run && return Ptr{Ptr{Cvoid}}(x + 72)
    f === :CreateSessionOptions && return Ptr{Ptr{Cvoid}}(x + 80)
    f === :SetOptimizedModelFilePath && return Ptr{Ptr{Cvoid}}(x + 88)
    f === :CloneSessionOptions && return Ptr{Ptr{Cvoid}}(x + 96)
    f === :SetSessionExecutionMode && return Ptr{Ptr{Cvoid}}(x + 104)
    f === :EnableProfiling && return Ptr{Ptr{Cvoid}}(x + 112)
    f === :DisableProfiling && return Ptr{Ptr{Cvoid}}(x + 120)
    f === :EnableMemPattern && return Ptr{Ptr{Cvoid}}(x + 128)
    f === :DisableMemPattern && return Ptr{Ptr{Cvoid}}(x + 136)
    f === :EnableCpuMemArena && return Ptr{Ptr{Cvoid}}(x + 144)
    f === :DisableCpuMemArena && return Ptr{Ptr{Cvoid}}(x + 152)
    f === :SetSessionLogId && return Ptr{Ptr{Cvoid}}(x + 160)
    f === :SetSessionLogVerbosityLevel && return Ptr{Ptr{Cvoid}}(x + 168)
    f === :SetSessionLogSeverityLevel && return Ptr{Ptr{Cvoid}}(x + 176)
    f === :SetSessionGraphOptimizationLevel && return Ptr{Ptr{Cvoid}}(x + 184)
    f === :SetIntraOpNumThreads && return Ptr{Ptr{Cvoid}}(x + 192)
    f === :SetInterOpNumThreads && return Ptr{Ptr{Cvoid}}(x + 200)
    f === :CreateCustomOpDomain && return Ptr{Ptr{Cvoid}}(x + 208)
    f === :CustomOpDomain_Add && return Ptr{Ptr{Cvoid}}(x + 216)
    f === :AddCustomOpDomain && return Ptr{Ptr{Cvoid}}(x + 224)
    f === :RegisterCustomOpsLibrary && return Ptr{Ptr{Cvoid}}(x + 232)
    f === :SessionGetInputCount && return Ptr{Ptr{Cvoid}}(x + 240)
    f === :SessionGetOutputCount && return Ptr{Ptr{Cvoid}}(x + 248)
    f === :SessionGetOverridableInitializerCount && return Ptr{Ptr{Cvoid}}(x + 256)
    f === :SessionGetInputTypeInfo && return Ptr{Ptr{Cvoid}}(x + 264)
    f === :SessionGetOutputTypeInfo && return Ptr{Ptr{Cvoid}}(x + 272)
    f === :SessionGetOverridableInitializerTypeInfo && return Ptr{Ptr{Cvoid}}(x + 280)
    f === :SessionGetInputName && return Ptr{Ptr{Cvoid}}(x + 288)
    f === :SessionGetOutputName && return Ptr{Ptr{Cvoid}}(x + 296)
    f === :SessionGetOverridableInitializerName && return Ptr{Ptr{Cvoid}}(x + 304)
    f === :CreateRunOptions && return Ptr{Ptr{Cvoid}}(x + 312)
    f === :RunOptionsSetRunLogVerbosityLevel && return Ptr{Ptr{Cvoid}}(x + 320)
    f === :RunOptionsSetRunLogSeverityLevel && return Ptr{Ptr{Cvoid}}(x + 328)
    f === :RunOptionsSetRunTag && return Ptr{Ptr{Cvoid}}(x + 336)
    f === :RunOptionsGetRunLogVerbosityLevel && return Ptr{Ptr{Cvoid}}(x + 344)
    f === :RunOptionsGetRunLogSeverityLevel && return Ptr{Ptr{Cvoid}}(x + 352)
    f === :RunOptionsGetRunTag && return Ptr{Ptr{Cvoid}}(x + 360)
    f === :RunOptionsSetTerminate && return Ptr{Ptr{Cvoid}}(x + 368)
    f === :RunOptionsUnsetTerminate && return Ptr{Ptr{Cvoid}}(x + 376)
    f === :CreateTensorAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 384)
    f === :CreateTensorWithDataAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 392)
    f === :IsTensor && return Ptr{Ptr{Cvoid}}(x + 400)
    f === :GetTensorMutableData && return Ptr{Ptr{Cvoid}}(x + 408)
    f === :FillStringTensor && return Ptr{Ptr{Cvoid}}(x + 416)
    f === :GetStringTensorDataLength && return Ptr{Ptr{Cvoid}}(x + 424)
    f === :GetStringTensorContent && return Ptr{Ptr{Cvoid}}(x + 432)
    f === :CastTypeInfoToTensorInfo && return Ptr{Ptr{Cvoid}}(x + 440)
    f === :GetOnnxTypeFromTypeInfo && return Ptr{Ptr{Cvoid}}(x + 448)
    f === :CreateTensorTypeAndShapeInfo && return Ptr{Ptr{Cvoid}}(x + 456)
    f === :SetTensorElementType && return Ptr{Ptr{Cvoid}}(x + 464)
    f === :SetDimensions && return Ptr{Ptr{Cvoid}}(x + 472)
    f === :GetTensorElementType && return Ptr{Ptr{Cvoid}}(x + 480)
    f === :GetDimensionsCount && return Ptr{Ptr{Cvoid}}(x + 488)
    f === :GetDimensions && return Ptr{Ptr{Cvoid}}(x + 496)
    f === :GetSymbolicDimensions && return Ptr{Ptr{Cvoid}}(x + 504)
    f === :GetTensorShapeElementCount && return Ptr{Ptr{Cvoid}}(x + 512)
    f === :GetTensorTypeAndShape && return Ptr{Ptr{Cvoid}}(x + 520)
    f === :GetTypeInfo && return Ptr{Ptr{Cvoid}}(x + 528)
    f === :GetValueType && return Ptr{Ptr{Cvoid}}(x + 536)
    f === :CreateMemoryInfo && return Ptr{Ptr{Cvoid}}(x + 544)
    f === :CreateCpuMemoryInfo && return Ptr{Ptr{Cvoid}}(x + 552)
    f === :CompareMemoryInfo && return Ptr{Ptr{Cvoid}}(x + 560)
    f === :MemoryInfoGetName && return Ptr{Ptr{Cvoid}}(x + 568)
    f === :MemoryInfoGetId && return Ptr{Ptr{Cvoid}}(x + 576)
    f === :MemoryInfoGetMemType && return Ptr{Ptr{Cvoid}}(x + 584)
    f === :MemoryInfoGetType && return Ptr{Ptr{Cvoid}}(x + 592)
    f === :AllocatorAlloc && return Ptr{Ptr{Cvoid}}(x + 600)
    f === :AllocatorFree && return Ptr{Ptr{Cvoid}}(x + 608)
    f === :AllocatorGetInfo && return Ptr{Ptr{Cvoid}}(x + 616)
    f === :GetAllocatorWithDefaultOptions && return Ptr{Ptr{Cvoid}}(x + 624)
    f === :AddFreeDimensionOverride && return Ptr{Ptr{Cvoid}}(x + 632)
    f === :GetValue && return Ptr{Ptr{Cvoid}}(x + 640)
    f === :GetValueCount && return Ptr{Ptr{Cvoid}}(x + 648)
    f === :CreateValue && return Ptr{Ptr{Cvoid}}(x + 656)
    f === :CreateOpaqueValue && return Ptr{Ptr{Cvoid}}(x + 664)
    f === :GetOpaqueValue && return Ptr{Ptr{Cvoid}}(x + 672)
    f === :KernelInfoGetAttribute_float && return Ptr{Ptr{Cvoid}}(x + 680)
    f === :KernelInfoGetAttribute_int64 && return Ptr{Ptr{Cvoid}}(x + 688)
    f === :KernelInfoGetAttribute_string && return Ptr{Ptr{Cvoid}}(x + 696)
    f === :KernelContext_GetInputCount && return Ptr{Ptr{Cvoid}}(x + 704)
    f === :KernelContext_GetOutputCount && return Ptr{Ptr{Cvoid}}(x + 712)
    f === :KernelContext_GetInput && return Ptr{Ptr{Cvoid}}(x + 720)
    f === :KernelContext_GetOutput && return Ptr{Ptr{Cvoid}}(x + 728)
    f === :ReleaseEnv && return Ptr{Ptr{Cvoid}}(x + 736)
    f === :ReleaseStatus && return Ptr{Ptr{Cvoid}}(x + 744)
    f === :ReleaseMemoryInfo && return Ptr{Ptr{Cvoid}}(x + 752)
    f === :ReleaseSession && return Ptr{Ptr{Cvoid}}(x + 760)
    f === :ReleaseValue && return Ptr{Ptr{Cvoid}}(x + 768)
    f === :ReleaseRunOptions && return Ptr{Ptr{Cvoid}}(x + 776)
    f === :ReleaseTypeInfo && return Ptr{Ptr{Cvoid}}(x + 784)
    f === :ReleaseTensorTypeAndShapeInfo && return Ptr{Ptr{Cvoid}}(x + 792)
    f === :ReleaseSessionOptions && return Ptr{Ptr{Cvoid}}(x + 800)
    f === :ReleaseCustomOpDomain && return Ptr{Ptr{Cvoid}}(x + 808)
    f === :GetDenotationFromTypeInfo && return Ptr{Ptr{Cvoid}}(x + 816)
    f === :CastTypeInfoToMapTypeInfo && return Ptr{Ptr{Cvoid}}(x + 824)
    f === :CastTypeInfoToSequenceTypeInfo && return Ptr{Ptr{Cvoid}}(x + 832)
    f === :GetMapKeyType && return Ptr{Ptr{Cvoid}}(x + 840)
    f === :GetMapValueType && return Ptr{Ptr{Cvoid}}(x + 848)
    f === :GetSequenceElementType && return Ptr{Ptr{Cvoid}}(x + 856)
    f === :ReleaseMapTypeInfo && return Ptr{Ptr{Cvoid}}(x + 864)
    f === :ReleaseSequenceTypeInfo && return Ptr{Ptr{Cvoid}}(x + 872)
    f === :SessionEndProfiling && return Ptr{Ptr{Cvoid}}(x + 880)
    f === :SessionGetModelMetadata && return Ptr{Ptr{Cvoid}}(x + 888)
    f === :ModelMetadataGetProducerName && return Ptr{Ptr{Cvoid}}(x + 896)
    f === :ModelMetadataGetGraphName && return Ptr{Ptr{Cvoid}}(x + 904)
    f === :ModelMetadataGetDomain && return Ptr{Ptr{Cvoid}}(x + 912)
    f === :ModelMetadataGetDescription && return Ptr{Ptr{Cvoid}}(x + 920)
    f === :ModelMetadataLookupCustomMetadataMap && return Ptr{Ptr{Cvoid}}(x + 928)
    f === :ModelMetadataGetVersion && return Ptr{Ptr{Cvoid}}(x + 936)
    f === :ReleaseModelMetadata && return Ptr{Ptr{Cvoid}}(x + 944)
    f === :CreateEnvWithGlobalThreadPools && return Ptr{Ptr{Cvoid}}(x + 952)
    f === :DisablePerSessionThreads && return Ptr{Ptr{Cvoid}}(x + 960)
    f === :CreateThreadingOptions && return Ptr{Ptr{Cvoid}}(x + 968)
    f === :ReleaseThreadingOptions && return Ptr{Ptr{Cvoid}}(x + 976)
    f === :ModelMetadataGetCustomMetadataMapKeys && return Ptr{Ptr{Cvoid}}(x + 984)
    f === :AddFreeDimensionOverrideByName && return Ptr{Ptr{Cvoid}}(x + 992)
    f === :GetAvailableProviders && return Ptr{Ptr{Cvoid}}(x + 1000)
    f === :ReleaseAvailableProviders && return Ptr{Ptr{Cvoid}}(x + 1008)
    f === :GetStringTensorElementLength && return Ptr{Ptr{Cvoid}}(x + 1016)
    f === :GetStringTensorElement && return Ptr{Ptr{Cvoid}}(x + 1024)
    f === :FillStringTensorElement && return Ptr{Ptr{Cvoid}}(x + 1032)
    f === :AddSessionConfigEntry && return Ptr{Ptr{Cvoid}}(x + 1040)
    f === :CreateAllocator && return Ptr{Ptr{Cvoid}}(x + 1048)
    f === :ReleaseAllocator && return Ptr{Ptr{Cvoid}}(x + 1056)
    f === :RunWithBinding && return Ptr{Ptr{Cvoid}}(x + 1064)
    f === :CreateIoBinding && return Ptr{Ptr{Cvoid}}(x + 1072)
    f === :ReleaseIoBinding && return Ptr{Ptr{Cvoid}}(x + 1080)
    f === :BindInput && return Ptr{Ptr{Cvoid}}(x + 1088)
    f === :BindOutput && return Ptr{Ptr{Cvoid}}(x + 1096)
    f === :BindOutputToDevice && return Ptr{Ptr{Cvoid}}(x + 1104)
    f === :GetBoundOutputNames && return Ptr{Ptr{Cvoid}}(x + 1112)
    f === :GetBoundOutputValues && return Ptr{Ptr{Cvoid}}(x + 1120)
    f === :ClearBoundInputs && return Ptr{Ptr{Cvoid}}(x + 1128)
    f === :ClearBoundOutputs && return Ptr{Ptr{Cvoid}}(x + 1136)
    f === :TensorAt && return Ptr{Ptr{Cvoid}}(x + 1144)
    f === :CreateAndRegisterAllocator && return Ptr{Ptr{Cvoid}}(x + 1152)
    f === :SetLanguageProjection && return Ptr{Ptr{Cvoid}}(x + 1160)
    f === :SessionGetProfilingStartTimeNs && return Ptr{Ptr{Cvoid}}(x + 1168)
    f === :SetGlobalIntraOpNumThreads && return Ptr{Ptr{Cvoid}}(x + 1176)
    f === :SetGlobalInterOpNumThreads && return Ptr{Ptr{Cvoid}}(x + 1184)
    f === :SetGlobalSpinControl && return Ptr{Ptr{Cvoid}}(x + 1192)
    f === :AddInitializer && return Ptr{Ptr{Cvoid}}(x + 1200)
    f === :CreateEnvWithCustomLoggerAndGlobalThreadPools && return Ptr{Ptr{Cvoid}}(x + 1208)
    f === :SessionOptionsAppendExecutionProvider_CUDA && return Ptr{Ptr{Cvoid}}(x + 1216)
    f === :SessionOptionsAppendExecutionProvider_ROCM && return Ptr{Ptr{Cvoid}}(x + 1224)
    f === :SessionOptionsAppendExecutionProvider_OpenVINO && return Ptr{Ptr{Cvoid}}(x + 1232)
    f === :SetGlobalDenormalAsZero && return Ptr{Ptr{Cvoid}}(x + 1240)
    f === :CreateArenaCfg && return Ptr{Ptr{Cvoid}}(x + 1248)
    f === :ReleaseArenaCfg && return Ptr{Ptr{Cvoid}}(x + 1256)
    f === :ModelMetadataGetGraphDescription && return Ptr{Ptr{Cvoid}}(x + 1264)
    f === :SessionOptionsAppendExecutionProvider_TensorRT && return Ptr{Ptr{Cvoid}}(x + 1272)
    f === :SetCurrentGpuDeviceId && return Ptr{Ptr{Cvoid}}(x + 1280)
    f === :GetCurrentGpuDeviceId && return Ptr{Ptr{Cvoid}}(x + 1288)
    f === :KernelInfoGetAttributeArray_float && return Ptr{Ptr{Cvoid}}(x + 1296)
    f === :KernelInfoGetAttributeArray_int64 && return Ptr{Ptr{Cvoid}}(x + 1304)
    f === :CreateArenaCfgV2 && return Ptr{Ptr{Cvoid}}(x + 1312)
    f === :AddRunConfigEntry && return Ptr{Ptr{Cvoid}}(x + 1320)
    f === :CreatePrepackedWeightsContainer && return Ptr{Ptr{Cvoid}}(x + 1328)
    f === :ReleasePrepackedWeightsContainer && return Ptr{Ptr{Cvoid}}(x + 1336)
    f === :CreateSessionWithPrepackedWeightsContainer && return Ptr{Ptr{Cvoid}}(x + 1344)
    f === :CreateSessionFromArrayWithPrepackedWeightsContainer && return Ptr{Ptr{Cvoid}}(x + 1352)
    f === :SessionOptionsAppendExecutionProvider_TensorRT_V2 && return Ptr{Ptr{Cvoid}}(x + 1360)
    f === :CreateTensorRTProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1368)
    f === :UpdateTensorRTProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1376)
    f === :GetTensorRTProviderOptionsAsString && return Ptr{Ptr{Cvoid}}(x + 1384)
    f === :ReleaseTensorRTProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1392)
    f === :EnableOrtCustomOps && return Ptr{Ptr{Cvoid}}(x + 1400)
    f === :RegisterAllocator && return Ptr{Ptr{Cvoid}}(x + 1408)
    f === :UnregisterAllocator && return Ptr{Ptr{Cvoid}}(x + 1416)
    f === :IsSparseTensor && return Ptr{Ptr{Cvoid}}(x + 1424)
    f === :CreateSparseTensorAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 1432)
    f === :FillSparseTensorCoo && return Ptr{Ptr{Cvoid}}(x + 1440)
    f === :FillSparseTensorCsr && return Ptr{Ptr{Cvoid}}(x + 1448)
    f === :FillSparseTensorBlockSparse && return Ptr{Ptr{Cvoid}}(x + 1456)
    f === :CreateSparseTensorWithValuesAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 1464)
    f === :UseCooIndices && return Ptr{Ptr{Cvoid}}(x + 1472)
    f === :UseCsrIndices && return Ptr{Ptr{Cvoid}}(x + 1480)
    f === :UseBlockSparseIndices && return Ptr{Ptr{Cvoid}}(x + 1488)
    f === :GetSparseTensorFormat && return Ptr{Ptr{Cvoid}}(x + 1496)
    f === :GetSparseTensorValuesTypeAndShape && return Ptr{Ptr{Cvoid}}(x + 1504)
    f === :GetSparseTensorValues && return Ptr{Ptr{Cvoid}}(x + 1512)
    f === :GetSparseTensorIndicesTypeShape && return Ptr{Ptr{Cvoid}}(x + 1520)
    f === :GetSparseTensorIndices && return Ptr{Ptr{Cvoid}}(x + 1528)
    f === :HasValue && return Ptr{Ptr{Cvoid}}(x + 1536)
    f === :KernelContext_GetGPUComputeStream && return Ptr{Ptr{Cvoid}}(x + 1544)
    f === :GetTensorMemoryInfo && return Ptr{Ptr{Cvoid}}(x + 1552)
    f === :GetExecutionProviderApi && return Ptr{Ptr{Cvoid}}(x + 1560)
    f === :SessionOptionsSetCustomCreateThreadFn && return Ptr{Ptr{Cvoid}}(x + 1568)
    f === :SessionOptionsSetCustomThreadCreationOptions && return Ptr{Ptr{Cvoid}}(x + 1576)
    f === :SessionOptionsSetCustomJoinThreadFn && return Ptr{Ptr{Cvoid}}(x + 1584)
    f === :SetGlobalCustomCreateThreadFn && return Ptr{Ptr{Cvoid}}(x + 1592)
    f === :SetGlobalCustomThreadCreationOptions && return Ptr{Ptr{Cvoid}}(x + 1600)
    f === :SetGlobalCustomJoinThreadFn && return Ptr{Ptr{Cvoid}}(x + 1608)
    f === :SynchronizeBoundInputs && return Ptr{Ptr{Cvoid}}(x + 1616)
    f === :SynchronizeBoundOutputs && return Ptr{Ptr{Cvoid}}(x + 1624)
    f === :SessionOptionsAppendExecutionProvider_CUDA_V2 && return Ptr{Ptr{Cvoid}}(x + 1632)
    f === :CreateCUDAProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1640)
    f === :UpdateCUDAProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1648)
    f === :GetCUDAProviderOptionsAsString && return Ptr{Ptr{Cvoid}}(x + 1656)
    f === :ReleaseCUDAProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1664)
    f === :SessionOptionsAppendExecutionProvider_MIGraphX && return Ptr{Ptr{Cvoid}}(x + 1672)
    f === :AddExternalInitializers && return Ptr{Ptr{Cvoid}}(x + 1680)
    f === :CreateOpAttr && return Ptr{Ptr{Cvoid}}(x + 1688)
    f === :ReleaseOpAttr && return Ptr{Ptr{Cvoid}}(x + 1696)
    f === :CreateOp && return Ptr{Ptr{Cvoid}}(x + 1704)
    f === :InvokeOp && return Ptr{Ptr{Cvoid}}(x + 1712)
    f === :ReleaseOp && return Ptr{Ptr{Cvoid}}(x + 1720)
    f === :SessionOptionsAppendExecutionProvider && return Ptr{Ptr{Cvoid}}(x + 1728)
    f === :CopyKernelInfo && return Ptr{Ptr{Cvoid}}(x + 1736)
    f === :ReleaseKernelInfo && return Ptr{Ptr{Cvoid}}(x + 1744)
    f === :GetTrainingApi && return Ptr{Ptr{Cvoid}}(x + 1752)
    f === :SessionOptionsAppendExecutionProvider_CANN && return Ptr{Ptr{Cvoid}}(x + 1760)
    f === :CreateCANNProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1768)
    f === :UpdateCANNProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1776)
    f === :GetCANNProviderOptionsAsString && return Ptr{Ptr{Cvoid}}(x + 1784)
    f === :ReleaseCANNProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1792)
    f === :MemoryInfoGetDeviceType && return Ptr{Ptr{Cvoid}}(x + 1800)
    f === :UpdateEnvWithCustomLogLevel && return Ptr{Ptr{Cvoid}}(x + 1808)
    f === :SetGlobalIntraOpThreadAffinity && return Ptr{Ptr{Cvoid}}(x + 1816)
    f === :RegisterCustomOpsLibrary_V2 && return Ptr{Ptr{Cvoid}}(x + 1824)
    f === :RegisterCustomOpsUsingFunction && return Ptr{Ptr{Cvoid}}(x + 1832)
    f === :KernelInfo_GetInputCount && return Ptr{Ptr{Cvoid}}(x + 1840)
    f === :KernelInfo_GetOutputCount && return Ptr{Ptr{Cvoid}}(x + 1848)
    f === :KernelInfo_GetInputName && return Ptr{Ptr{Cvoid}}(x + 1856)
    f === :KernelInfo_GetOutputName && return Ptr{Ptr{Cvoid}}(x + 1864)
    f === :KernelInfo_GetInputTypeInfo && return Ptr{Ptr{Cvoid}}(x + 1872)
    f === :KernelInfo_GetOutputTypeInfo && return Ptr{Ptr{Cvoid}}(x + 1880)
    f === :KernelInfoGetAttribute_tensor && return Ptr{Ptr{Cvoid}}(x + 1888)
    f === :HasSessionConfigEntry && return Ptr{Ptr{Cvoid}}(x + 1896)
    f === :GetSessionConfigEntry && return Ptr{Ptr{Cvoid}}(x + 1904)
    f === :SessionOptionsAppendExecutionProvider_Dnnl && return Ptr{Ptr{Cvoid}}(x + 1912)
    f === :CreateDnnlProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1920)
    f === :UpdateDnnlProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1928)
    f === :GetDnnlProviderOptionsAsString && return Ptr{Ptr{Cvoid}}(x + 1936)
    f === :ReleaseDnnlProviderOptions && return Ptr{Ptr{Cvoid}}(x + 1944)
    f === :KernelInfo_GetNodeName && return Ptr{Ptr{Cvoid}}(x + 1952)
    f === :KernelInfo_GetLogger && return Ptr{Ptr{Cvoid}}(x + 1960)
    f === :KernelContext_GetLogger && return Ptr{Ptr{Cvoid}}(x + 1968)
    f === :Logger_LogMessage && return Ptr{Ptr{Cvoid}}(x + 1976)
    f === :Logger_GetLoggingSeverityLevel && return Ptr{Ptr{Cvoid}}(x + 1984)
    f === :KernelInfoGetConstantInput_tensor && return Ptr{Ptr{Cvoid}}(x + 1992)
    f === :CastTypeInfoToOptionalTypeInfo && return Ptr{Ptr{Cvoid}}(x + 2000)
    f === :GetOptionalContainedTypeInfo && return Ptr{Ptr{Cvoid}}(x + 2008)
    f === :GetResizedStringTensorElementBuffer && return Ptr{Ptr{Cvoid}}(x + 2016)
    f === :KernelContext_GetAllocator && return Ptr{Ptr{Cvoid}}(x + 2024)
    f === :GetBuildInfoString && return Ptr{Ptr{Cvoid}}(x + 2032)
    f === :CreateROCMProviderOptions && return Ptr{Ptr{Cvoid}}(x + 2040)
    f === :UpdateROCMProviderOptions && return Ptr{Ptr{Cvoid}}(x + 2048)
    f === :GetROCMProviderOptionsAsString && return Ptr{Ptr{Cvoid}}(x + 2056)
    f === :ReleaseROCMProviderOptions && return Ptr{Ptr{Cvoid}}(x + 2064)
    f === :CreateAndRegisterAllocatorV2 && return Ptr{Ptr{Cvoid}}(x + 2072)
    f === :RunAsync && return Ptr{Ptr{Cvoid}}(x + 2080)
    f === :UpdateTensorRTProviderOptionsWithValue && return Ptr{Ptr{Cvoid}}(x + 2088)
    f === :GetTensorRTProviderOptionsByName && return Ptr{Ptr{Cvoid}}(x + 2096)
    f === :UpdateCUDAProviderOptionsWithValue && return Ptr{Ptr{Cvoid}}(x + 2104)
    f === :GetCUDAProviderOptionsByName && return Ptr{Ptr{Cvoid}}(x + 2112)
    f === :KernelContext_GetResource && return Ptr{Ptr{Cvoid}}(x + 2120)
    f === :SetUserLoggingFunction && return Ptr{Ptr{Cvoid}}(x + 2128)
    f === :ShapeInferContext_GetInputCount && return Ptr{Ptr{Cvoid}}(x + 2136)
    f === :ShapeInferContext_GetInputTypeShape && return Ptr{Ptr{Cvoid}}(x + 2144)
    f === :ShapeInferContext_GetAttribute && return Ptr{Ptr{Cvoid}}(x + 2152)
    f === :ShapeInferContext_SetOutputTypeShape && return Ptr{Ptr{Cvoid}}(x + 2160)
    f === :SetSymbolicDimensions && return Ptr{Ptr{Cvoid}}(x + 2168)
    f === :ReadOpAttr && return Ptr{Ptr{Cvoid}}(x + 2176)
    f === :SetDeterministicCompute && return Ptr{Ptr{Cvoid}}(x + 2184)
    f === :KernelContext_ParallelFor && return Ptr{Ptr{Cvoid}}(x + 2192)
    f === :SessionOptionsAppendExecutionProvider_OpenVINO_V2 && return Ptr{Ptr{Cvoid}}(x + 2200)
    f === :SessionOptionsAppendExecutionProvider_VitisAI && return Ptr{Ptr{Cvoid}}(x + 2208)
    f === :KernelContext_GetScratchBuffer && return Ptr{Ptr{Cvoid}}(x + 2216)
    f === :KernelInfoGetAllocator && return Ptr{Ptr{Cvoid}}(x + 2224)
    f === :AddExternalInitializersFromFilesInMemory && return Ptr{Ptr{Cvoid}}(x + 2232)
    f === :CreateLoraAdapter && return Ptr{Ptr{Cvoid}}(x + 2240)
    f === :CreateLoraAdapterFromArray && return Ptr{Ptr{Cvoid}}(x + 2248)
    f === :ReleaseLoraAdapter && return Ptr{Ptr{Cvoid}}(x + 2256)
    f === :RunOptionsAddActiveLoraAdapter && return Ptr{Ptr{Cvoid}}(x + 2264)
    f === :SetEpDynamicOptions && return Ptr{Ptr{Cvoid}}(x + 2272)
    f === :ReleaseValueInfo && return Ptr{Ptr{Cvoid}}(x + 2280)
    f === :ReleaseNode && return Ptr{Ptr{Cvoid}}(x + 2288)
    f === :ReleaseGraph && return Ptr{Ptr{Cvoid}}(x + 2296)
    f === :ReleaseModel && return Ptr{Ptr{Cvoid}}(x + 2304)
    f === :GetValueInfoName && return Ptr{Ptr{Cvoid}}(x + 2312)
    f === :GetValueInfoTypeInfo && return Ptr{Ptr{Cvoid}}(x + 2320)
    f === :GetModelEditorApi && return Ptr{Ptr{Cvoid}}(x + 2328)
    f === :CreateTensorWithDataAndDeleterAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 2336)
    f === :SessionOptionsSetLoadCancellationFlag && return Ptr{Ptr{Cvoid}}(x + 2344)
    f === :GetCompileApi && return Ptr{Ptr{Cvoid}}(x + 2352)
    f === :CreateKeyValuePairs && return Ptr{Ptr{Cvoid}}(x + 2360)
    f === :AddKeyValuePair && return Ptr{Ptr{Cvoid}}(x + 2368)
    f === :GetKeyValue && return Ptr{Ptr{Cvoid}}(x + 2376)
    f === :GetKeyValuePairs && return Ptr{Ptr{Cvoid}}(x + 2384)
    f === :RemoveKeyValuePair && return Ptr{Ptr{Cvoid}}(x + 2392)
    f === :ReleaseKeyValuePairs && return Ptr{Ptr{Cvoid}}(x + 2400)
    f === :RegisterExecutionProviderLibrary && return Ptr{Ptr{Cvoid}}(x + 2408)
    f === :UnregisterExecutionProviderLibrary && return Ptr{Ptr{Cvoid}}(x + 2416)
    f === :GetEpDevices && return Ptr{Ptr{Cvoid}}(x + 2424)
    f === :SessionOptionsAppendExecutionProvider_V2 && return Ptr{Ptr{Cvoid}}(x + 2432)
    f === :SessionOptionsSetEpSelectionPolicy && return Ptr{Ptr{Cvoid}}(x + 2440)
    f === :SessionOptionsSetEpSelectionPolicyDelegate && return Ptr{Ptr{Cvoid}}(x + 2448)
    f === :HardwareDevice_Type && return Ptr{Ptr{Cvoid}}(x + 2456)
    f === :HardwareDevice_VendorId && return Ptr{Ptr{Cvoid}}(x + 2464)
    f === :HardwareDevice_Vendor && return Ptr{Ptr{Cvoid}}(x + 2472)
    f === :HardwareDevice_DeviceId && return Ptr{Ptr{Cvoid}}(x + 2480)
    f === :HardwareDevice_Metadata && return Ptr{Ptr{Cvoid}}(x + 2488)
    f === :EpDevice_EpName && return Ptr{Ptr{Cvoid}}(x + 2496)
    f === :EpDevice_EpVendor && return Ptr{Ptr{Cvoid}}(x + 2504)
    f === :EpDevice_EpMetadata && return Ptr{Ptr{Cvoid}}(x + 2512)
    f === :EpDevice_EpOptions && return Ptr{Ptr{Cvoid}}(x + 2520)
    f === :EpDevice_Device && return Ptr{Ptr{Cvoid}}(x + 2528)
    f === :GetEpApi && return Ptr{Ptr{Cvoid}}(x + 2536)
    f === :GetTensorSizeInBytes && return Ptr{Ptr{Cvoid}}(x + 2544)
    f === :AllocatorGetStats && return Ptr{Ptr{Cvoid}}(x + 2552)
    f === :CreateMemoryInfo_V2 && return Ptr{Ptr{Cvoid}}(x + 2560)
    f === :MemoryInfoGetDeviceMemType && return Ptr{Ptr{Cvoid}}(x + 2568)
    f === :MemoryInfoGetVendorId && return Ptr{Ptr{Cvoid}}(x + 2576)
    f === :ValueInfo_GetValueProducer && return Ptr{Ptr{Cvoid}}(x + 2584)
    f === :ValueInfo_GetValueNumConsumers && return Ptr{Ptr{Cvoid}}(x + 2592)
    f === :ValueInfo_GetValueConsumers && return Ptr{Ptr{Cvoid}}(x + 2600)
    f === :ValueInfo_GetInitializerValue && return Ptr{Ptr{Cvoid}}(x + 2608)
    f === :ValueInfo_GetExternalInitializerInfo && return Ptr{Ptr{Cvoid}}(x + 2616)
    f === :ValueInfo_IsRequiredGraphInput && return Ptr{Ptr{Cvoid}}(x + 2624)
    f === :ValueInfo_IsOptionalGraphInput && return Ptr{Ptr{Cvoid}}(x + 2632)
    f === :ValueInfo_IsGraphOutput && return Ptr{Ptr{Cvoid}}(x + 2640)
    f === :ValueInfo_IsConstantInitializer && return Ptr{Ptr{Cvoid}}(x + 2648)
    f === :ValueInfo_IsFromOuterScope && return Ptr{Ptr{Cvoid}}(x + 2656)
    f === :Graph_GetName && return Ptr{Ptr{Cvoid}}(x + 2664)
    f === :Graph_GetModelPath && return Ptr{Ptr{Cvoid}}(x + 2672)
    f === :Graph_GetOnnxIRVersion && return Ptr{Ptr{Cvoid}}(x + 2680)
    f === :Graph_GetNumOperatorSets && return Ptr{Ptr{Cvoid}}(x + 2688)
    f === :Graph_GetOperatorSets && return Ptr{Ptr{Cvoid}}(x + 2696)
    f === :Graph_GetNumInputs && return Ptr{Ptr{Cvoid}}(x + 2704)
    f === :Graph_GetInputs && return Ptr{Ptr{Cvoid}}(x + 2712)
    f === :Graph_GetNumOutputs && return Ptr{Ptr{Cvoid}}(x + 2720)
    f === :Graph_GetOutputs && return Ptr{Ptr{Cvoid}}(x + 2728)
    f === :Graph_GetNumInitializers && return Ptr{Ptr{Cvoid}}(x + 2736)
    f === :Graph_GetInitializers && return Ptr{Ptr{Cvoid}}(x + 2744)
    f === :Graph_GetNumNodes && return Ptr{Ptr{Cvoid}}(x + 2752)
    f === :Graph_GetNodes && return Ptr{Ptr{Cvoid}}(x + 2760)
    f === :Graph_GetParentNode && return Ptr{Ptr{Cvoid}}(x + 2768)
    f === :Graph_GetGraphView && return Ptr{Ptr{Cvoid}}(x + 2776)
    f === :Node_GetId && return Ptr{Ptr{Cvoid}}(x + 2784)
    f === :Node_GetName && return Ptr{Ptr{Cvoid}}(x + 2792)
    f === :Node_GetOperatorType && return Ptr{Ptr{Cvoid}}(x + 2800)
    f === :Node_GetDomain && return Ptr{Ptr{Cvoid}}(x + 2808)
    f === :Node_GetSinceVersion && return Ptr{Ptr{Cvoid}}(x + 2816)
    f === :Node_GetNumInputs && return Ptr{Ptr{Cvoid}}(x + 2824)
    f === :Node_GetInputs && return Ptr{Ptr{Cvoid}}(x + 2832)
    f === :Node_GetNumOutputs && return Ptr{Ptr{Cvoid}}(x + 2840)
    f === :Node_GetOutputs && return Ptr{Ptr{Cvoid}}(x + 2848)
    f === :Node_GetNumImplicitInputs && return Ptr{Ptr{Cvoid}}(x + 2856)
    f === :Node_GetImplicitInputs && return Ptr{Ptr{Cvoid}}(x + 2864)
    f === :Node_GetNumAttributes && return Ptr{Ptr{Cvoid}}(x + 2872)
    f === :Node_GetAttributes && return Ptr{Ptr{Cvoid}}(x + 2880)
    f === :Node_GetAttributeByName && return Ptr{Ptr{Cvoid}}(x + 2888)
    f === :OpAttr_GetTensorAttributeAsOrtValue && return Ptr{Ptr{Cvoid}}(x + 2896)
    f === :OpAttr_GetType && return Ptr{Ptr{Cvoid}}(x + 2904)
    f === :OpAttr_GetName && return Ptr{Ptr{Cvoid}}(x + 2912)
    f === :Node_GetNumSubgraphs && return Ptr{Ptr{Cvoid}}(x + 2920)
    f === :Node_GetSubgraphs && return Ptr{Ptr{Cvoid}}(x + 2928)
    f === :Node_GetGraph && return Ptr{Ptr{Cvoid}}(x + 2936)
    f === :Node_GetEpName && return Ptr{Ptr{Cvoid}}(x + 2944)
    f === :ReleaseExternalInitializerInfo && return Ptr{Ptr{Cvoid}}(x + 2952)
    f === :ExternalInitializerInfo_GetFilePath && return Ptr{Ptr{Cvoid}}(x + 2960)
    f === :ExternalInitializerInfo_GetFileOffset && return Ptr{Ptr{Cvoid}}(x + 2968)
    f === :ExternalInitializerInfo_GetByteSize && return Ptr{Ptr{Cvoid}}(x + 2976)
    f === :GetRunConfigEntry && return Ptr{Ptr{Cvoid}}(x + 2984)
    f === :EpDevice_MemoryInfo && return Ptr{Ptr{Cvoid}}(x + 2992)
    f === :CreateSharedAllocator && return Ptr{Ptr{Cvoid}}(x + 3000)
    f === :GetSharedAllocator && return Ptr{Ptr{Cvoid}}(x + 3008)
    f === :ReleaseSharedAllocator && return Ptr{Ptr{Cvoid}}(x + 3016)
    f === :GetTensorData && return Ptr{Ptr{Cvoid}}(x + 3024)
    f === :GetSessionOptionsConfigEntries && return Ptr{Ptr{Cvoid}}(x + 3032)
    f === :SessionGetMemoryInfoForInputs && return Ptr{Ptr{Cvoid}}(x + 3040)
    f === :SessionGetMemoryInfoForOutputs && return Ptr{Ptr{Cvoid}}(x + 3048)
    f === :SessionGetEpDeviceForInputs && return Ptr{Ptr{Cvoid}}(x + 3056)
    f === :CreateSyncStreamForEpDevice && return Ptr{Ptr{Cvoid}}(x + 3064)
    f === :SyncStream_GetHandle && return Ptr{Ptr{Cvoid}}(x + 3072)
    f === :ReleaseSyncStream && return Ptr{Ptr{Cvoid}}(x + 3080)
    f === :CopyTensors && return Ptr{Ptr{Cvoid}}(x + 3088)
    f === :Graph_GetModelMetadata && return Ptr{Ptr{Cvoid}}(x + 3096)
    f === :GetModelCompatibilityForEpDevices && return Ptr{Ptr{Cvoid}}(x + 3104)
    f === :CreateExternalInitializerInfo && return Ptr{Ptr{Cvoid}}(x + 3112)
    return getfield(x, f)
end

function Base.getproperty(x::OrtApi, f::Symbol)
    r = Ref{OrtApi}(x)
    ptr = Base.unsafe_convert(Ptr{OrtApi}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{OrtApi}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function Base.propertynames(x::OrtApi, private::Bool = false)
    (:CreateStatus, :GetErrorCode, :GetErrorMessage, :CreateEnv, :CreateEnvWithCustomLogger, :EnableTelemetryEvents, :DisableTelemetryEvents, :CreateSession, :CreateSessionFromArray, :Run, :CreateSessionOptions, :SetOptimizedModelFilePath, :CloneSessionOptions, :SetSessionExecutionMode, :EnableProfiling, :DisableProfiling, :EnableMemPattern, :DisableMemPattern, :EnableCpuMemArena, :DisableCpuMemArena, :SetSessionLogId, :SetSessionLogVerbosityLevel, :SetSessionLogSeverityLevel, :SetSessionGraphOptimizationLevel, :SetIntraOpNumThreads, :SetInterOpNumThreads, :CreateCustomOpDomain, :CustomOpDomain_Add, :AddCustomOpDomain, :RegisterCustomOpsLibrary, :SessionGetInputCount, :SessionGetOutputCount, :SessionGetOverridableInitializerCount, :SessionGetInputTypeInfo, :SessionGetOutputTypeInfo, :SessionGetOverridableInitializerTypeInfo, :SessionGetInputName, :SessionGetOutputName, :SessionGetOverridableInitializerName, :CreateRunOptions, :RunOptionsSetRunLogVerbosityLevel, :RunOptionsSetRunLogSeverityLevel, :RunOptionsSetRunTag, :RunOptionsGetRunLogVerbosityLevel, :RunOptionsGetRunLogSeverityLevel, :RunOptionsGetRunTag, :RunOptionsSetTerminate, :RunOptionsUnsetTerminate, :CreateTensorAsOrtValue, :CreateTensorWithDataAsOrtValue, :IsTensor, :GetTensorMutableData, :FillStringTensor, :GetStringTensorDataLength, :GetStringTensorContent, :CastTypeInfoToTensorInfo, :GetOnnxTypeFromTypeInfo, :CreateTensorTypeAndShapeInfo, :SetTensorElementType, :SetDimensions, :GetTensorElementType, :GetDimensionsCount, :GetDimensions, :GetSymbolicDimensions, :GetTensorShapeElementCount, :GetTensorTypeAndShape, :GetTypeInfo, :GetValueType, :CreateMemoryInfo, :CreateCpuMemoryInfo, :CompareMemoryInfo, :MemoryInfoGetName, :MemoryInfoGetId, :MemoryInfoGetMemType, :MemoryInfoGetType, :AllocatorAlloc, :AllocatorFree, :AllocatorGetInfo, :GetAllocatorWithDefaultOptions, :AddFreeDimensionOverride, :GetValue, :GetValueCount, :CreateValue, :CreateOpaqueValue, :GetOpaqueValue, :KernelInfoGetAttribute_float, :KernelInfoGetAttribute_int64, :KernelInfoGetAttribute_string, :KernelContext_GetInputCount, :KernelContext_GetOutputCount, :KernelContext_GetInput, :KernelContext_GetOutput, :ReleaseEnv, :ReleaseStatus, :ReleaseMemoryInfo, :ReleaseSession, :ReleaseValue, :ReleaseRunOptions, :ReleaseTypeInfo, :ReleaseTensorTypeAndShapeInfo, :ReleaseSessionOptions, :ReleaseCustomOpDomain, :GetDenotationFromTypeInfo, :CastTypeInfoToMapTypeInfo, :CastTypeInfoToSequenceTypeInfo, :GetMapKeyType, :GetMapValueType, :GetSequenceElementType, :ReleaseMapTypeInfo, :ReleaseSequenceTypeInfo, :SessionEndProfiling, :SessionGetModelMetadata, :ModelMetadataGetProducerName, :ModelMetadataGetGraphName, :ModelMetadataGetDomain, :ModelMetadataGetDescription, :ModelMetadataLookupCustomMetadataMap, :ModelMetadataGetVersion, :ReleaseModelMetadata, :CreateEnvWithGlobalThreadPools, :DisablePerSessionThreads, :CreateThreadingOptions, :ReleaseThreadingOptions, :ModelMetadataGetCustomMetadataMapKeys, :AddFreeDimensionOverrideByName, :GetAvailableProviders, :ReleaseAvailableProviders, :GetStringTensorElementLength, :GetStringTensorElement, :FillStringTensorElement, :AddSessionConfigEntry, :CreateAllocator, :ReleaseAllocator, :RunWithBinding, :CreateIoBinding, :ReleaseIoBinding, :BindInput, :BindOutput, :BindOutputToDevice, :GetBoundOutputNames, :GetBoundOutputValues, :ClearBoundInputs, :ClearBoundOutputs, :TensorAt, :CreateAndRegisterAllocator, :SetLanguageProjection, :SessionGetProfilingStartTimeNs, :SetGlobalIntraOpNumThreads, :SetGlobalInterOpNumThreads, :SetGlobalSpinControl, :AddInitializer, :CreateEnvWithCustomLoggerAndGlobalThreadPools, :SessionOptionsAppendExecutionProvider_CUDA, :SessionOptionsAppendExecutionProvider_ROCM, :SessionOptionsAppendExecutionProvider_OpenVINO, :SetGlobalDenormalAsZero, :CreateArenaCfg, :ReleaseArenaCfg, :ModelMetadataGetGraphDescription, :SessionOptionsAppendExecutionProvider_TensorRT, :SetCurrentGpuDeviceId, :GetCurrentGpuDeviceId, :KernelInfoGetAttributeArray_float, :KernelInfoGetAttributeArray_int64, :CreateArenaCfgV2, :AddRunConfigEntry, :CreatePrepackedWeightsContainer, :ReleasePrepackedWeightsContainer, :CreateSessionWithPrepackedWeightsContainer, :CreateSessionFromArrayWithPrepackedWeightsContainer, :SessionOptionsAppendExecutionProvider_TensorRT_V2, :CreateTensorRTProviderOptions, :UpdateTensorRTProviderOptions, :GetTensorRTProviderOptionsAsString, :ReleaseTensorRTProviderOptions, :EnableOrtCustomOps, :RegisterAllocator, :UnregisterAllocator, :IsSparseTensor, :CreateSparseTensorAsOrtValue, :FillSparseTensorCoo, :FillSparseTensorCsr, :FillSparseTensorBlockSparse, :CreateSparseTensorWithValuesAsOrtValue, :UseCooIndices, :UseCsrIndices, :UseBlockSparseIndices, :GetSparseTensorFormat, :GetSparseTensorValuesTypeAndShape, :GetSparseTensorValues, :GetSparseTensorIndicesTypeShape, :GetSparseTensorIndices, :HasValue, :KernelContext_GetGPUComputeStream, :GetTensorMemoryInfo, :GetExecutionProviderApi, :SessionOptionsSetCustomCreateThreadFn, :SessionOptionsSetCustomThreadCreationOptions, :SessionOptionsSetCustomJoinThreadFn, :SetGlobalCustomCreateThreadFn, :SetGlobalCustomThreadCreationOptions, :SetGlobalCustomJoinThreadFn, :SynchronizeBoundInputs, :SynchronizeBoundOutputs, :SessionOptionsAppendExecutionProvider_CUDA_V2, :CreateCUDAProviderOptions, :UpdateCUDAProviderOptions, :GetCUDAProviderOptionsAsString, :ReleaseCUDAProviderOptions, :SessionOptionsAppendExecutionProvider_MIGraphX, :AddExternalInitializers, :CreateOpAttr, :ReleaseOpAttr, :CreateOp, :InvokeOp, :ReleaseOp, :SessionOptionsAppendExecutionProvider, :CopyKernelInfo, :ReleaseKernelInfo, :GetTrainingApi, :SessionOptionsAppendExecutionProvider_CANN, :CreateCANNProviderOptions, :UpdateCANNProviderOptions, :GetCANNProviderOptionsAsString, :ReleaseCANNProviderOptions, :MemoryInfoGetDeviceType, :UpdateEnvWithCustomLogLevel, :SetGlobalIntraOpThreadAffinity, :RegisterCustomOpsLibrary_V2, :RegisterCustomOpsUsingFunction, :KernelInfo_GetInputCount, :KernelInfo_GetOutputCount, :KernelInfo_GetInputName, :KernelInfo_GetOutputName, :KernelInfo_GetInputTypeInfo, :KernelInfo_GetOutputTypeInfo, :KernelInfoGetAttribute_tensor, :HasSessionConfigEntry, :GetSessionConfigEntry, :SessionOptionsAppendExecutionProvider_Dnnl, :CreateDnnlProviderOptions, :UpdateDnnlProviderOptions, :GetDnnlProviderOptionsAsString, :ReleaseDnnlProviderOptions, :KernelInfo_GetNodeName, :KernelInfo_GetLogger, :KernelContext_GetLogger, :Logger_LogMessage, :Logger_GetLoggingSeverityLevel, :KernelInfoGetConstantInput_tensor, :CastTypeInfoToOptionalTypeInfo, :GetOptionalContainedTypeInfo, :GetResizedStringTensorElementBuffer, :KernelContext_GetAllocator, :GetBuildInfoString, :CreateROCMProviderOptions, :UpdateROCMProviderOptions, :GetROCMProviderOptionsAsString, :ReleaseROCMProviderOptions, :CreateAndRegisterAllocatorV2, :RunAsync, :UpdateTensorRTProviderOptionsWithValue, :GetTensorRTProviderOptionsByName, :UpdateCUDAProviderOptionsWithValue, :GetCUDAProviderOptionsByName, :KernelContext_GetResource, :SetUserLoggingFunction, :ShapeInferContext_GetInputCount, :ShapeInferContext_GetInputTypeShape, :ShapeInferContext_GetAttribute, :ShapeInferContext_SetOutputTypeShape, :SetSymbolicDimensions, :ReadOpAttr, :SetDeterministicCompute, :KernelContext_ParallelFor, :SessionOptionsAppendExecutionProvider_OpenVINO_V2, :SessionOptionsAppendExecutionProvider_VitisAI, :KernelContext_GetScratchBuffer, :KernelInfoGetAllocator, :AddExternalInitializersFromFilesInMemory, :CreateLoraAdapter, :CreateLoraAdapterFromArray, :ReleaseLoraAdapter, :RunOptionsAddActiveLoraAdapter, :SetEpDynamicOptions, :ReleaseValueInfo, :ReleaseNode, :ReleaseGraph, :ReleaseModel, :GetValueInfoName, :GetValueInfoTypeInfo, :GetModelEditorApi, :CreateTensorWithDataAndDeleterAsOrtValue, :SessionOptionsSetLoadCancellationFlag, :GetCompileApi, :CreateKeyValuePairs, :AddKeyValuePair, :GetKeyValue, :GetKeyValuePairs, :RemoveKeyValuePair, :ReleaseKeyValuePairs, :RegisterExecutionProviderLibrary, :UnregisterExecutionProviderLibrary, :GetEpDevices, :SessionOptionsAppendExecutionProvider_V2, :SessionOptionsSetEpSelectionPolicy, :SessionOptionsSetEpSelectionPolicyDelegate, :HardwareDevice_Type, :HardwareDevice_VendorId, :HardwareDevice_Vendor, :HardwareDevice_DeviceId, :HardwareDevice_Metadata, :EpDevice_EpName, :EpDevice_EpVendor, :EpDevice_EpMetadata, :EpDevice_EpOptions, :EpDevice_Device, :GetEpApi, :GetTensorSizeInBytes, :AllocatorGetStats, :CreateMemoryInfo_V2, :MemoryInfoGetDeviceMemType, :MemoryInfoGetVendorId, :ValueInfo_GetValueProducer, :ValueInfo_GetValueNumConsumers, :ValueInfo_GetValueConsumers, :ValueInfo_GetInitializerValue, :ValueInfo_GetExternalInitializerInfo, :ValueInfo_IsRequiredGraphInput, :ValueInfo_IsOptionalGraphInput, :ValueInfo_IsGraphOutput, :ValueInfo_IsConstantInitializer, :ValueInfo_IsFromOuterScope, :Graph_GetName, :Graph_GetModelPath, :Graph_GetOnnxIRVersion, :Graph_GetNumOperatorSets, :Graph_GetOperatorSets, :Graph_GetNumInputs, :Graph_GetInputs, :Graph_GetNumOutputs, :Graph_GetOutputs, :Graph_GetNumInitializers, :Graph_GetInitializers, :Graph_GetNumNodes, :Graph_GetNodes, :Graph_GetParentNode, :Graph_GetGraphView, :Node_GetId, :Node_GetName, :Node_GetOperatorType, :Node_GetDomain, :Node_GetSinceVersion, :Node_GetNumInputs, :Node_GetInputs, :Node_GetNumOutputs, :Node_GetOutputs, :Node_GetNumImplicitInputs, :Node_GetImplicitInputs, :Node_GetNumAttributes, :Node_GetAttributes, :Node_GetAttributeByName, :OpAttr_GetTensorAttributeAsOrtValue, :OpAttr_GetType, :OpAttr_GetName, :Node_GetNumSubgraphs, :Node_GetSubgraphs, :Node_GetGraph, :Node_GetEpName, :ReleaseExternalInitializerInfo, :ExternalInitializerInfo_GetFilePath, :ExternalInitializerInfo_GetFileOffset, :ExternalInitializerInfo_GetByteSize, :GetRunConfigEntry, :EpDevice_MemoryInfo, :CreateSharedAllocator, :GetSharedAllocator, :ReleaseSharedAllocator, :GetTensorData, :GetSessionOptionsConfigEntries, :SessionGetMemoryInfoForInputs, :SessionGetMemoryInfoForOutputs, :SessionGetEpDeviceForInputs, :CreateSyncStreamForEpDevice, :SyncStream_GetHandle, :ReleaseSyncStream, :CopyTensors, :Graph_GetModelMetadata, :GetModelCompatibilityForEpDevices, :CreateExternalInitializerInfo, if private
            fieldnames(typeof(x))
        else
            ()
        end...)
end

mutable struct OrtTrainingApi end

struct OrtModelEditorApi
    CreateTensorTypeInfo::Ptr{Cvoid}
    CreateSparseTensorTypeInfo::Ptr{Cvoid}
    CreateMapTypeInfo::Ptr{Cvoid}
    CreateSequenceTypeInfo::Ptr{Cvoid}
    CreateOptionalTypeInfo::Ptr{Cvoid}
    CreateValueInfo::Ptr{Cvoid}
    CreateNode::Ptr{Cvoid}
    CreateGraph::Ptr{Cvoid}
    SetGraphInputs::Ptr{Cvoid}
    SetGraphOutputs::Ptr{Cvoid}
    AddInitializerToGraph::Ptr{Cvoid}
    AddNodeToGraph::Ptr{Cvoid}
    CreateModel::Ptr{Cvoid}
    AddGraphToModel::Ptr{Cvoid}
    CreateSessionFromModel::Ptr{Cvoid}
    CreateModelEditorSession::Ptr{Cvoid}
    CreateModelEditorSessionFromArray::Ptr{Cvoid}
    SessionGetOpsetForDomain::Ptr{Cvoid}
    ApplyModelToModelEditorSession::Ptr{Cvoid}
    FinalizeModelEditorSession::Ptr{Cvoid}
end

struct OrtCompileApi
    ReleaseModelCompilationOptions::Ptr{Cvoid}
    CreateModelCompilationOptionsFromSessionOptions::Ptr{Cvoid}
    ModelCompilationOptions_SetInputModelPath::Ptr{Cvoid}
    ModelCompilationOptions_SetInputModelFromBuffer::Ptr{Cvoid}
    ModelCompilationOptions_SetOutputModelPath::Ptr{Cvoid}
    ModelCompilationOptions_SetOutputModelExternalInitializersFile::Ptr{Cvoid}
    ModelCompilationOptions_SetOutputModelBuffer::Ptr{Cvoid}
    ModelCompilationOptions_SetEpContextEmbedMode::Ptr{Cvoid}
    CompileModel::Ptr{Cvoid}
    ModelCompilationOptions_SetFlags::Ptr{Cvoid}
    ModelCompilationOptions_SetEpContextBinaryInformation::Ptr{Cvoid}
    ModelCompilationOptions_SetGraphOptimizationLevel::Ptr{Cvoid}
    ModelCompilationOptions_SetOutputModelWriteFunc::Ptr{Cvoid}
    ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc::Ptr{Cvoid}
end

struct OrtEpApi
    CreateEpDevice::Ptr{Cvoid}
    ReleaseEpDevice::Ptr{Cvoid}
    EpGraphSupportInfo_AddNodesToFuse::Ptr{Cvoid}
    EpGraphSupportInfo_AddSingleNode::Ptr{Cvoid}
    NodeComputeContext_NodeName::Ptr{Cvoid}
    EpDevice_AddAllocatorInfo::Ptr{Cvoid}
    MemoryInfo_GetMemoryDevice::Ptr{Cvoid}
    Value_GetMemoryDevice::Ptr{Cvoid}
    MemoryDevice_AreEqual::Ptr{Cvoid}
    MemoryDevice_GetDeviceType::Ptr{Cvoid}
    MemoryDevice_GetMemoryType::Ptr{Cvoid}
    MemoryDevice_GetVendorId::Ptr{Cvoid}
    MemoryDevice_GetDeviceId::Ptr{Cvoid}
    SyncStream_GetImpl::Ptr{Cvoid}
    SyncStream_GetSyncId::Ptr{Cvoid}
    GetSyncIdForLastWaitOnSyncStream::Ptr{Cvoid}
end

struct OrtApiBase
    GetApi::Ptr{Cvoid}
    GetVersionString::Ptr{Cvoid}
end

function OrtGetApiBase()
    @ccall OnnxRuntime.OrtGetApiBase()::Ptr{OrtApiBase}
end

# typedef void ( * OrtThreadWorkerFn ) ( void * ort_worker_fn_param )
const OrtThreadWorkerFn = Ptr{Cvoid}

struct OrtCustomHandleType
    __place_holder::Cchar
end

const OrtCustomThreadHandle = Ptr{OrtCustomHandleType}

# typedef OrtCustomThreadHandle ( * OrtCustomCreateThreadFn ) ( void * ort_custom_thread_creation_options , OrtThreadWorkerFn ort_thread_worker_fn , void * ort_worker_fn_param )
const OrtCustomCreateThreadFn = Ptr{Cvoid}

# typedef void ( * OrtCustomJoinThreadFn ) ( OrtCustomThreadHandle ort_custom_thread_handle )
const OrtCustomJoinThreadFn = Ptr{Cvoid}

# typedef OrtStatus * ( ORT_API_CALL * RegisterCustomOpsFn
const RegisterCustomOpsFn = Ptr{Cvoid}

# typedef void ( * RunAsyncCallbackFn ) ( void * user_data , OrtValue * * outputs , size_t num_outputs , OrtStatusPtr status )
const RunAsyncCallbackFn = Ptr{Cvoid}

@cenum OrtCompiledModelCompatibility::UInt32 begin
    OrtCompiledModelCompatibility_EP_NOT_APPLICABLE = 0
    OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL = 1
    OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION = 2
    OrtCompiledModelCompatibility_EP_UNSUPPORTED = 3
end

@cenum OrtCustomOpInputOutputCharacteristic::UInt32 begin
    INPUT_OUTPUT_REQUIRED = 0
    INPUT_OUTPUT_OPTIONAL = 1
    INPUT_OUTPUT_VARIADIC = 2
end

@cenum OrtCompileApiFlags::UInt32 begin
    OrtCompileApiFlags_NONE = 0
    OrtCompileApiFlags_ERROR_IF_NO_NODES_COMPILED = 1
    OrtCompileApiFlags_ERROR_IF_OUTPUT_FILE_EXISTS = 2
end

function OrtSessionOptionsAppendExecutionProvider_CUDA(options, device_id)
    @ccall OnnxRuntime.OrtSessionOptionsAppendExecutionProvider_CUDA(options::Ptr{OrtSessionOptions}, device_id::Cint)::OrtStatusPtr
end

function OrtSessionOptionsAppendExecutionProvider_ROCM(options, device_id)
    @ccall OnnxRuntime.OrtSessionOptionsAppendExecutionProvider_ROCM(options::Ptr{OrtSessionOptions}, device_id::Cint)::OrtStatusPtr
end

function OrtSessionOptionsAppendExecutionProvider_MIGraphX(options, device_id)
    @ccall OnnxRuntime.OrtSessionOptionsAppendExecutionProvider_MIGraphX(options::Ptr{OrtSessionOptions}, device_id::Cint)::OrtStatusPtr
end

function OrtSessionOptionsAppendExecutionProvider_Dnnl(options, use_arena)
    @ccall OnnxRuntime.OrtSessionOptionsAppendExecutionProvider_Dnnl(options::Ptr{OrtSessionOptions}, use_arena::Cint)::OrtStatusPtr
end

function OrtSessionOptionsAppendExecutionProvider_Tensorrt(options, device_id)
    @ccall OnnxRuntime.OrtSessionOptionsAppendExecutionProvider_Tensorrt(options::Ptr{OrtSessionOptions}, device_id::Cint)::OrtStatusPtr
end

struct OrtEp
    ort_version_supported::UInt32
    GetName::Ptr{Cvoid}
    GetCapability::Ptr{Cvoid}
    Compile::Ptr{Cvoid}
    ReleaseNodeComputeInfos::Ptr{Cvoid}
    GetPreferredDataLayout::Ptr{Cvoid}
    ShouldConvertDataLayoutForOp::Ptr{Cvoid}
    SetDynamicOptions::Ptr{Cvoid}
    OnRunStart::Ptr{Cvoid}
    OnRunEnd::Ptr{Cvoid}
    CreateAllocator::Ptr{Cvoid}
    CreateSyncStreamForDevice::Ptr{Cvoid}
    GetCompiledModelCompatibilityInfo::Ptr{Cvoid}
end

struct OrtEpFactory
    ort_version_supported::UInt32
    GetName::Ptr{Cvoid}
    GetVendor::Ptr{Cvoid}
    GetSupportedDevices::Ptr{Cvoid}
    CreateEp::Ptr{Cvoid}
    ReleaseEp::Ptr{Cvoid}
    GetVendorId::Ptr{Cvoid}
    GetVersion::Ptr{Cvoid}
    ValidateCompiledModelCompatibilityInfo::Ptr{Cvoid}
    CreateAllocator::Ptr{Cvoid}
    ReleaseAllocator::Ptr{Cvoid}
    CreateDataTransfer::Ptr{Cvoid}
    IsStreamAware::Ptr{Cvoid}
    CreateSyncStreamForDevice::Ptr{Cvoid}
end

mutable struct OrtEpGraphSupportInfo end

mutable struct OrtMemoryDevice end

mutable struct OrtNodeComputeContext end

struct OrtDataTransferImpl
    ort_version_supported::UInt32
    Release::Ptr{Cvoid}
    CanCopy::Ptr{Cvoid}
    CopyTensors::Ptr{Cvoid}
end

struct OrtSyncNotificationImpl
    ort_version_supported::UInt32
    Release::Ptr{Cvoid}
    Activate::Ptr{Cvoid}
    WaitOnDevice::Ptr{Cvoid}
    WaitOnHost::Ptr{Cvoid}
end

struct OrtSyncStreamImpl
    ort_version_supported::UInt32
    Release::Ptr{Cvoid}
    GetHandle::Ptr{Cvoid}
    CreateNotification::Ptr{Cvoid}
    Flush::Ptr{Cvoid}
    OnSessionRunEnd::Ptr{Cvoid}
end

struct OrtNodeFusionOptions
    ort_version_supported::UInt32
    drop_constant_initializers::Bool
end

struct OrtNodeComputeInfo
    ort_version_supported::UInt32
    CreateState::Ptr{Cvoid}
    Compute::Ptr{Cvoid}
    ReleaseState::Ptr{Cvoid}
end

@cenum OrtEpDataLayout::UInt32 begin
    OrtEpDataLayout_NCHW = 0
    OrtEpDataLayout_NHWC = 1
    OrtEpDataLayout_Default = 0
end

# typedef OrtStatus * ( * CreateEpApiFactoriesFn ) ( _In_ const char * registered_name , _In_ const OrtApiBase * ort_api_base , _In_ const OrtLogger * default_logger , _Inout_ OrtEpFactory * * factories , _In_ size_t max_factories , _Out_ size_t * num_factories )
const CreateEpApiFactoriesFn = Ptr{Cvoid}

# typedef OrtStatus * ( * ReleaseEpApiFactoryFn ) ( _In_ OrtEpFactory * factory )
const ReleaseEpApiFactoryFn = Ptr{Cvoid}

const ORT_API_VERSION = 23

# Skipping MacroDefinition: ORT_ALL_ARGS_NONNULL __attribute__ ( ( nonnull ) )

CreateStatus(apis::OrtApi, code, msg) = ccall(Base.getproperty(apis, :CreateStatus), Ptr{OrtStatus}, (OrtErrorCode, Ptr{Cchar}), code, msg)
GetErrorCode(apis::OrtApi, status) = ccall(Base.getproperty(apis, :GetErrorCode), OrtErrorCode, (Ptr{OrtStatus}), status)
GetErrorMessage(apis::OrtApi, status) = ccall(Base.getproperty(apis, :GetErrorMessage), Ptr{Cchar}, (Ptr{OrtStatus}), status)
CreateEnv(apis::OrtApi, log_severity_level, logid, out) = ccall(Base.getproperty(apis, :CreateEnv), OrtStatusPtr, (OrtLoggingLevel, Ptr{Cchar}, Ptr{Ptr{OrtEnv}}), log_severity_level, logid, out)
CreateEnvWithCustomLogger(apis::OrtApi, logging_function, logger_param, log_severity_level, logid, out) = ccall(Base.getproperty(apis, :CreateEnvWithCustomLogger), OrtStatusPtr, (OrtLoggingFunction, Ptr{Cvoid}, OrtLoggingLevel, Ptr{Cchar}, Ptr{Ptr{OrtEnv}}), logging_function, logger_param, log_severity_level, logid, out)
EnableTelemetryEvents(apis::OrtApi, env) = ccall(Base.getproperty(apis, :EnableTelemetryEvents), OrtStatusPtr, (Ptr{OrtEnv}), env)
DisableTelemetryEvents(apis::OrtApi, env) = ccall(Base.getproperty(apis, :DisableTelemetryEvents), OrtStatusPtr, (Ptr{OrtEnv}), env)
CreateSession(apis::OrtApi, env, model_path, options, out) = ccall(Base.getproperty(apis, :CreateSession), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{wchar_t}, Ptr{OrtSessionOptions}, Ptr{Ptr{OrtSession}}), env, model_path, options, out)
CreateSessionFromArray(apis::OrtApi, env, model_data, model_data_length, options, out) = ccall(Base.getproperty(apis, :CreateSessionFromArray), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Cvoid}, size_t, Ptr{OrtSessionOptions}, Ptr{Ptr{OrtSession}}), env, model_data, model_data_length, options, out)
Run(apis::OrtApi, session, run_options, input_names, inputs, input_len, output_names, output_names_len, outputs) = ccall(Base.getproperty(apis, :Run), OrtStatusPtr, (Ptr{OrtSession}, Ptr{OrtRunOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{OrtValue}}, size_t, Ptr{Ptr{Cchar}}, size_t, Ptr{Ptr{OrtValue}}), session, run_options, input_names, inputs, input_len, output_names, output_names_len, outputs)
CreateSessionOptions(apis::OrtApi, options) = ccall(Base.getproperty(apis, :CreateSessionOptions), OrtStatusPtr, (Ptr{Ptr{OrtSessionOptions}}), options)
SetOptimizedModelFilePath(apis::OrtApi, options, optimized_model_filepath) = ccall(Base.getproperty(apis, :SetOptimizedModelFilePath), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{wchar_t}), options, optimized_model_filepath)
CloneSessionOptions(apis::OrtApi, in_options, out_options) = ccall(Base.getproperty(apis, :CloneSessionOptions), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{OrtSessionOptions}}), in_options, out_options)
SetSessionExecutionMode(apis::OrtApi, options, execution_mode) = ccall(Base.getproperty(apis, :SetSessionExecutionMode), OrtStatusPtr, (Ptr{OrtSessionOptions}, ExecutionMode), options, execution_mode)
EnableProfiling(apis::OrtApi, options, profile_file_prefix) = ccall(Base.getproperty(apis, :EnableProfiling), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{wchar_t}), options, profile_file_prefix)
DisableProfiling(apis::OrtApi, options) = ccall(Base.getproperty(apis, :DisableProfiling), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
EnableMemPattern(apis::OrtApi, options) = ccall(Base.getproperty(apis, :EnableMemPattern), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
DisableMemPattern(apis::OrtApi, options) = ccall(Base.getproperty(apis, :DisableMemPattern), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
EnableCpuMemArena(apis::OrtApi, options) = ccall(Base.getproperty(apis, :EnableCpuMemArena), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
DisableCpuMemArena(apis::OrtApi, options) = ccall(Base.getproperty(apis, :DisableCpuMemArena), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
SetSessionLogId(apis::OrtApi, options, logid) = ccall(Base.getproperty(apis, :SetSessionLogId), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}), options, logid)
SetSessionLogVerbosityLevel(apis::OrtApi, options, session_log_verbosity_level) = ccall(Base.getproperty(apis, :SetSessionLogVerbosityLevel), OrtStatusPtr, (Ptr{OrtSessionOptions}, Cint), options, session_log_verbosity_level)
SetSessionLogSeverityLevel(apis::OrtApi, options, session_log_severity_level) = ccall(Base.getproperty(apis, :SetSessionLogSeverityLevel), OrtStatusPtr, (Ptr{OrtSessionOptions}, Cint), options, session_log_severity_level)
SetSessionGraphOptimizationLevel(apis::OrtApi, options, graph_optimization_level) = ccall(Base.getproperty(apis, :SetSessionGraphOptimizationLevel), OrtStatusPtr, (Ptr{OrtSessionOptions}, GraphOptimizationLevel), options, graph_optimization_level)
SetIntraOpNumThreads(apis::OrtApi, options, intra_op_num_threads) = ccall(Base.getproperty(apis, :SetIntraOpNumThreads), OrtStatusPtr, (Ptr{OrtSessionOptions}, Cint), options, intra_op_num_threads)
SetInterOpNumThreads(apis::OrtApi, options, inter_op_num_threads) = ccall(Base.getproperty(apis, :SetInterOpNumThreads), OrtStatusPtr, (Ptr{OrtSessionOptions}, Cint), options, inter_op_num_threads)
CreateCustomOpDomain(apis::OrtApi, domain, out) = ccall(Base.getproperty(apis, :CreateCustomOpDomain), OrtStatusPtr, (Ptr{Cchar}, Ptr{Ptr{OrtCustomOpDomain}}), domain, out)
CustomOpDomain_Add(apis::OrtApi, custom_op_domain, op) = ccall(Base.getproperty(apis, :CustomOpDomain_Add), OrtStatusPtr, (Ptr{OrtCustomOpDomain}, Ptr{OrtCustomOp}), custom_op_domain, op)
AddCustomOpDomain(apis::OrtApi, options, custom_op_domain) = ccall(Base.getproperty(apis, :AddCustomOpDomain), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtCustomOpDomain}), options, custom_op_domain)
RegisterCustomOpsLibrary(apis::OrtApi, options, library_path, library_handle) = ccall(Base.getproperty(apis, :RegisterCustomOpsLibrary), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}), options, library_path, library_handle)
SessionGetInputCount(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :SessionGetInputCount), OrtStatusPtr, (Ptr{OrtSession}, Ptr{size_t}), session, out)
SessionGetOutputCount(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :SessionGetOutputCount), OrtStatusPtr, (Ptr{OrtSession}, Ptr{size_t}), session, out)
SessionGetOverridableInitializerCount(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :SessionGetOverridableInitializerCount), OrtStatusPtr, (Ptr{OrtSession}, Ptr{size_t}), session, out)
SessionGetInputTypeInfo(apis::OrtApi, session, index, type_info) = ccall(Base.getproperty(apis, :SessionGetInputTypeInfo), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{Ptr{OrtTypeInfo}}), session, index, type_info)
SessionGetOutputTypeInfo(apis::OrtApi, session, index, type_info) = ccall(Base.getproperty(apis, :SessionGetOutputTypeInfo), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{Ptr{OrtTypeInfo}}), session, index, type_info)
SessionGetOverridableInitializerTypeInfo(apis::OrtApi, session, index, type_info) = ccall(Base.getproperty(apis, :SessionGetOverridableInitializerTypeInfo), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{Ptr{OrtTypeInfo}}), session, index, type_info)
SessionGetInputName(apis::OrtApi, session, index, allocator, value) = ccall(Base.getproperty(apis, :SessionGetInputName), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), session, index, allocator, value)
SessionGetOutputName(apis::OrtApi, session, index, allocator, value) = ccall(Base.getproperty(apis, :SessionGetOutputName), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), session, index, allocator, value)
SessionGetOverridableInitializerName(apis::OrtApi, session, index, allocator, value) = ccall(Base.getproperty(apis, :SessionGetOverridableInitializerName), OrtStatusPtr, (Ptr{OrtSession}, size_t, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), session, index, allocator, value)
CreateRunOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateRunOptions), OrtStatusPtr, (Ptr{Ptr{OrtRunOptions}}), out)
RunOptionsSetRunLogVerbosityLevel(apis::OrtApi, options, log_verbosity_level) = ccall(Base.getproperty(apis, :RunOptionsSetRunLogVerbosityLevel), OrtStatusPtr, (Ptr{OrtRunOptions}, Cint), options, log_verbosity_level)
RunOptionsSetRunLogSeverityLevel(apis::OrtApi, options, log_severity_level) = ccall(Base.getproperty(apis, :RunOptionsSetRunLogSeverityLevel), OrtStatusPtr, (Ptr{OrtRunOptions}, Cint), options, log_severity_level)
RunOptionsSetRunTag(apis::OrtApi, options, run_tag) = ccall(Base.getproperty(apis, :RunOptionsSetRunTag), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{Cchar}), options, run_tag)
RunOptionsGetRunLogVerbosityLevel(apis::OrtApi, options, log_verbosity_level) = ccall(Base.getproperty(apis, :RunOptionsGetRunLogVerbosityLevel), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{Cint}), options, log_verbosity_level)
RunOptionsGetRunLogSeverityLevel(apis::OrtApi, options, log_severity_level) = ccall(Base.getproperty(apis, :RunOptionsGetRunLogSeverityLevel), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{Cint}), options, log_severity_level)
RunOptionsGetRunTag(apis::OrtApi, options, run_tag) = ccall(Base.getproperty(apis, :RunOptionsGetRunTag), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{Ptr{Cchar}}), options, run_tag)
RunOptionsSetTerminate(apis::OrtApi, options) = ccall(Base.getproperty(apis, :RunOptionsSetTerminate), OrtStatusPtr, (Ptr{OrtRunOptions}), options)
RunOptionsUnsetTerminate(apis::OrtApi, options) = ccall(Base.getproperty(apis, :RunOptionsUnsetTerminate), OrtStatusPtr, (Ptr{OrtRunOptions}), options)
CreateTensorAsOrtValue(apis::OrtApi, allocator, shape, shape_len, type, out) = ccall(Base.getproperty(apis, :CreateTensorAsOrtValue), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{int64_t}, size_t, ONNXTensorElementDataType, Ptr{Ptr{OrtValue}}), allocator, shape, shape_len, type, out)
CreateTensorWithDataAsOrtValue(apis::OrtApi, info, p_data, p_data_len, shape, shape_len, type, out) = ccall(Base.getproperty(apis, :CreateTensorWithDataAsOrtValue), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{Cvoid}, size_t, Ptr{int64_t}, size_t, ONNXTensorElementDataType, Ptr{Ptr{OrtValue}}), info, p_data, p_data_len, shape, shape_len, type, out)
IsTensor(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :IsTensor), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cint}), value, out)
GetTensorMutableData(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetTensorMutableData), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{Cvoid}}), value, out)
FillStringTensor(apis::OrtApi, value, s, s_len) = ccall(Base.getproperty(apis, :FillStringTensor), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{Cchar}}, size_t), value, s, s_len)
GetStringTensorDataLength(apis::OrtApi, value, len) = ccall(Base.getproperty(apis, :GetStringTensorDataLength), OrtStatusPtr, (Ptr{OrtValue}, Ptr{size_t}), value, len)
GetStringTensorContent(apis::OrtApi, value, s, s_len, offsets, offsets_len) = ccall(Base.getproperty(apis, :GetStringTensorContent), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cvoid}, size_t, Ptr{size_t}, size_t), value, s, s_len, offsets, offsets_len)
CastTypeInfoToTensorInfo(apis::OrtApi, type_info, out) = ccall(Base.getproperty(apis, :CastTypeInfoToTensorInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), type_info, out)
GetOnnxTypeFromTypeInfo(apis::OrtApi, type_info, out) = ccall(Base.getproperty(apis, :GetOnnxTypeFromTypeInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Cvoid}), type_info, out)
CreateTensorTypeAndShapeInfo(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateTensorTypeAndShapeInfo), OrtStatusPtr, (Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), out)
SetTensorElementType(apis::OrtApi, info, type) = ccall(Base.getproperty(apis, :SetTensorElementType), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Cvoid), info, type)
SetDimensions(apis::OrtApi, info, dim_values, dim_count) = ccall(Base.getproperty(apis, :SetDimensions), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{int64_t}, size_t), info, dim_values, dim_count)
GetTensorElementType(apis::OrtApi, info, out) = ccall(Base.getproperty(apis, :GetTensorElementType), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{Cvoid}), info, out)
GetDimensionsCount(apis::OrtApi, info, out) = ccall(Base.getproperty(apis, :GetDimensionsCount), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{size_t}), info, out)
GetDimensions(apis::OrtApi, info, dim_values, dim_values_length) = ccall(Base.getproperty(apis, :GetDimensions), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{int64_t}, size_t), info, dim_values, dim_values_length)
GetSymbolicDimensions(apis::OrtApi, info, dim_params, dim_params_length) = ccall(Base.getproperty(apis, :GetSymbolicDimensions), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{Ptr{Cchar}}, size_t), info, dim_params, dim_params_length)
GetTensorShapeElementCount(apis::OrtApi, info, out) = ccall(Base.getproperty(apis, :GetTensorShapeElementCount), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{size_t}), info, out)
GetTensorTypeAndShape(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetTensorTypeAndShape), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), value, out)
GetTypeInfo(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetTypeInfo), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{OrtTypeInfo}}), value, out)
GetValueType(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetValueType), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cvoid}), value, out)
CreateMemoryInfo(apis::OrtApi, name, type, id, mem_type, out) = ccall(Base.getproperty(apis, :CreateMemoryInfo), OrtStatusPtr, (Ptr{Cchar}, Cvoid, Cint, Cvoid, Ptr{Ptr{OrtMemoryInfo}}), name, type, id, mem_type, out)
CreateCpuMemoryInfo(apis::OrtApi, type, mem_type, out) = ccall(Base.getproperty(apis, :CreateCpuMemoryInfo), OrtStatusPtr, (Cvoid, Cvoid, Ptr{Ptr{OrtMemoryInfo}}), type, mem_type, out)
CompareMemoryInfo(apis::OrtApi, info1, info2, out) = ccall(Base.getproperty(apis, :CompareMemoryInfo), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{OrtMemoryInfo}, Ptr{Cint}), info1, info2, out)
MemoryInfoGetName(apis::OrtApi, ptr, out) = ccall(Base.getproperty(apis, :MemoryInfoGetName), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{Ptr{Cchar}}), ptr, out)
MemoryInfoGetId(apis::OrtApi, ptr, out) = ccall(Base.getproperty(apis, :MemoryInfoGetId), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{Cint}), ptr, out)
MemoryInfoGetMemType(apis::OrtApi, ptr, out) = ccall(Base.getproperty(apis, :MemoryInfoGetMemType), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{OrtMemType}), ptr, out)
MemoryInfoGetType(apis::OrtApi, ptr, out) = ccall(Base.getproperty(apis, :MemoryInfoGetType), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{OrtAllocatorType}), ptr, out)
AllocatorAlloc(apis::OrtApi, ort_allocator, size, out) = ccall(Base.getproperty(apis, :AllocatorAlloc), OrtStatusPtr, (Ptr{OrtAllocator}, size_t, Ptr{Ptr{Cvoid}}), ort_allocator, size, out)
AllocatorFree(apis::OrtApi, ort_allocator, p) = ccall(Base.getproperty(apis, :AllocatorFree), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{Cvoid}), ort_allocator, p)
AllocatorGetInfo(apis::OrtApi, ort_allocator, out) = ccall(Base.getproperty(apis, :AllocatorGetInfo), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{Ptr{Cvoid}}), ort_allocator, out)
GetAllocatorWithDefaultOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :GetAllocatorWithDefaultOptions), OrtStatusPtr, (Ptr{Ptr{OrtAllocator}}), out)
AddFreeDimensionOverride(apis::OrtApi, options, dim_denotation, dim_value) = ccall(Base.getproperty(apis, :AddFreeDimensionOverride), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, int64_t), options, dim_denotation, dim_value)
GetValue(apis::OrtApi, value, index, allocator, out) = ccall(Base.getproperty(apis, :GetValue), OrtStatusPtr, (Ptr{OrtValue}, Cint, Ptr{OrtAllocator}, Ptr{Ptr{OrtValue}}), value, index, allocator, out)
GetValueCount(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetValueCount), OrtStatusPtr, (Ptr{OrtValue}, Ptr{size_t}), value, out)
CreateValue(apis::OrtApi, in, num_values, value_type, out) = ccall(Base.getproperty(apis, :CreateValue), OrtStatusPtr, (Ptr{Ptr{OrtValue}}, size_t, Cvoid, Ptr{Ptr{OrtValue}}), in, num_values, value_type, out)
CreateOpaqueValue(apis::OrtApi, domain_name, type_name, data_container, data_container_size, out) = ccall(Base.getproperty(apis, :CreateOpaqueValue), OrtStatusPtr, (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cvoid}, size_t, Ptr{Ptr{OrtValue}}), domain_name, type_name, data_container, data_container_size, out)
GetOpaqueValue(apis::OrtApi, domain_name, type_name, in, data_container, data_container_size) = ccall(Base.getproperty(apis, :GetOpaqueValue), OrtStatusPtr, (Ptr{Cchar}, Ptr{Cchar}, Ptr{OrtValue}, Ptr{Cvoid}, size_t), domain_name, type_name, in, data_container, data_container_size)
KernelInfoGetAttribute_float(apis::OrtApi, info, name, out) = ccall(Base.getproperty(apis, :KernelInfoGetAttribute_float), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{Cfloat}), info, name, out)
KernelInfoGetAttribute_int64(apis::OrtApi, info, name, out) = ccall(Base.getproperty(apis, :KernelInfoGetAttribute_int64), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{int64_t}), info, name, out)
KernelInfoGetAttribute_string(apis::OrtApi, info, name, out, size) = ccall(Base.getproperty(apis, :KernelInfoGetAttribute_string), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{Cchar}, Ptr{size_t}), info, name, out, size)
KernelContext_GetInputCount(apis::OrtApi, context, out) = ccall(Base.getproperty(apis, :KernelContext_GetInputCount), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{size_t}), context, out)
KernelContext_GetOutputCount(apis::OrtApi, context, out) = ccall(Base.getproperty(apis, :KernelContext_GetOutputCount), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{size_t}), context, out)
KernelContext_GetInput(apis::OrtApi, context, index, out) = ccall(Base.getproperty(apis, :KernelContext_GetInput), OrtStatusPtr, (Ptr{OrtKernelContext}, size_t, Ptr{Ptr{OrtValue}}), context, index, out)
KernelContext_GetOutput(apis::OrtApi, context, index, dim_values, dim_count, out) = ccall(Base.getproperty(apis, :KernelContext_GetOutput), OrtStatusPtr, (Ptr{OrtKernelContext}, size_t, Ptr{int64_t}, size_t, Ptr{Ptr{OrtValue}}), context, index, dim_values, dim_count, out)
ReleaseEnv(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseEnv), Cvoid, (Ptr{OrtEnv}), input)
ReleaseStatus(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseStatus), Cvoid, (Ptr{OrtStatus}), input)
ReleaseMemoryInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseMemoryInfo), Cvoid, (Ptr{OrtMemoryInfo}), input)
ReleaseSession(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseSession), Cvoid, (Ptr{OrtSession}), input)
ReleaseValue(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseValue), Cvoid, (Ptr{OrtValue}), input)
ReleaseRunOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseRunOptions), Cvoid, (Ptr{OrtRunOptions}), input)
ReleaseTypeInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseTypeInfo), Cvoid, (Ptr{OrtTypeInfo}), input)
ReleaseTensorTypeAndShapeInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseTensorTypeAndShapeInfo), Cvoid, (Ptr{OrtTensorTypeAndShapeInfo}), input)
ReleaseSessionOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseSessionOptions), Cvoid, (Ptr{OrtSessionOptions}), input)
ReleaseCustomOpDomain(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseCustomOpDomain), Cvoid, (Ptr{OrtCustomOpDomain}), input)
GetDenotationFromTypeInfo(apis::OrtApi, type_info, denotation, len) = ccall(Base.getproperty(apis, :GetDenotationFromTypeInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Ptr{Cchar}}, Ptr{size_t}), type_info, denotation, len)
CastTypeInfoToMapTypeInfo(apis::OrtApi, type_info, out) = ccall(Base.getproperty(apis, :CastTypeInfoToMapTypeInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Ptr{OrtMapTypeInfo}}), type_info, out)
CastTypeInfoToSequenceTypeInfo(apis::OrtApi, type_info, out) = ccall(Base.getproperty(apis, :CastTypeInfoToSequenceTypeInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Ptr{OrtSequenceTypeInfo}}), type_info, out)
GetMapKeyType(apis::OrtApi, map_type_info, out) = ccall(Base.getproperty(apis, :GetMapKeyType), OrtStatusPtr, (Ptr{OrtMapTypeInfo}, Ptr{Cvoid}), map_type_info, out)
GetMapValueType(apis::OrtApi, map_type_info, type_info) = ccall(Base.getproperty(apis, :GetMapValueType), OrtStatusPtr, (Ptr{OrtMapTypeInfo}, Ptr{Ptr{OrtTypeInfo}}), map_type_info, type_info)
GetSequenceElementType(apis::OrtApi, sequence_type_info, type_info) = ccall(Base.getproperty(apis, :GetSequenceElementType), OrtStatusPtr, (Ptr{OrtSequenceTypeInfo}, Ptr{Ptr{OrtTypeInfo}}), sequence_type_info, type_info)
ReleaseMapTypeInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseMapTypeInfo), Cvoid, (Ptr{OrtMapTypeInfo}), input)
ReleaseSequenceTypeInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseSequenceTypeInfo), Cvoid, (Ptr{OrtSequenceTypeInfo}), input)
SessionEndProfiling(apis::OrtApi, session, allocator, out) = ccall(Base.getproperty(apis, :SessionEndProfiling), OrtStatusPtr, (Ptr{OrtSession}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), session, allocator, out)
SessionGetModelMetadata(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :SessionGetModelMetadata), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{OrtModelMetadata}}), session, out)
ModelMetadataGetProducerName(apis::OrtApi, model_metadata, allocator, value) = ccall(Base.getproperty(apis, :ModelMetadataGetProducerName), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), model_metadata, allocator, value)
ModelMetadataGetGraphName(apis::OrtApi, model_metadata, allocator, value) = ccall(Base.getproperty(apis, :ModelMetadataGetGraphName), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), model_metadata, allocator, value)
ModelMetadataGetDomain(apis::OrtApi, model_metadata, allocator, value) = ccall(Base.getproperty(apis, :ModelMetadataGetDomain), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), model_metadata, allocator, value)
ModelMetadataGetDescription(apis::OrtApi, model_metadata, allocator, value) = ccall(Base.getproperty(apis, :ModelMetadataGetDescription), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), model_metadata, allocator, value)
ModelMetadataLookupCustomMetadataMap(apis::OrtApi, model_metadata, allocator, key, value) = ccall(Base.getproperty(apis, :ModelMetadataLookupCustomMetadataMap), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Cchar}, Ptr{Ptr{Cchar}}), model_metadata, allocator, key, value)
ModelMetadataGetVersion(apis::OrtApi, model_metadata, value) = ccall(Base.getproperty(apis, :ModelMetadataGetVersion), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{int64_t}), model_metadata, value)
ReleaseModelMetadata(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseModelMetadata), Cvoid, (Ptr{OrtModelMetadata}), input)
CreateEnvWithGlobalThreadPools(apis::OrtApi, log_severity_level, logid, tp_options, out) = ccall(Base.getproperty(apis, :CreateEnvWithGlobalThreadPools), OrtStatusPtr, (OrtLoggingLevel, Ptr{Cchar}, Ptr{OrtThreadingOptions}, Ptr{Ptr{OrtEnv}}), log_severity_level, logid, tp_options, out)
DisablePerSessionThreads(apis::OrtApi, options) = ccall(Base.getproperty(apis, :DisablePerSessionThreads), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
CreateThreadingOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateThreadingOptions), OrtStatusPtr, (Ptr{Ptr{OrtThreadingOptions}}), out)
ReleaseThreadingOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseThreadingOptions), Cvoid, (Ptr{OrtThreadingOptions}), input)
ModelMetadataGetCustomMetadataMapKeys(apis::OrtApi, model_metadata, allocator, keys, num_keys) = ccall(Base.getproperty(apis, :ModelMetadataGetCustomMetadataMapKeys), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{int64_t}), model_metadata, allocator, keys, num_keys)
AddFreeDimensionOverrideByName(apis::OrtApi, options, dim_name, dim_value) = ccall(Base.getproperty(apis, :AddFreeDimensionOverrideByName), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, int64_t), options, dim_name, dim_value)
GetAvailableProviders(apis::OrtApi, out_ptr, provider_length) = ccall(Base.getproperty(apis, :GetAvailableProviders), OrtStatusPtr, (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Cint}), out_ptr, provider_length)
ReleaseAvailableProviders(apis::OrtApi, ptr, providers_length) = ccall(Base.getproperty(apis, :ReleaseAvailableProviders), OrtStatusPtr, (Ptr{Ptr{Cchar}}, Cint), ptr, providers_length)
GetStringTensorElementLength(apis::OrtApi, value, index, out) = ccall(Base.getproperty(apis, :GetStringTensorElementLength), OrtStatusPtr, (Ptr{OrtValue}, size_t, Ptr{size_t}), value, index, out)
GetStringTensorElement(apis::OrtApi, value, s_len, index, s) = ccall(Base.getproperty(apis, :GetStringTensorElement), OrtStatusPtr, (Ptr{OrtValue}, size_t, size_t, Ptr{Cvoid}), value, s_len, index, s)
FillStringTensorElement(apis::OrtApi, value, s, index) = ccall(Base.getproperty(apis, :FillStringTensorElement), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cchar}, size_t), value, s, index)
AddSessionConfigEntry(apis::OrtApi, options, config_key, config_value) = ccall(Base.getproperty(apis, :AddSessionConfigEntry), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{Cchar}), options, config_key, config_value)
CreateAllocator(apis::OrtApi, session, mem_info, out) = ccall(Base.getproperty(apis, :CreateAllocator), OrtStatusPtr, (Ptr{OrtSession}, Ptr{OrtMemoryInfo}, Ptr{Ptr{OrtAllocator}}), session, mem_info, out)
ReleaseAllocator(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseAllocator), Cvoid, (Ptr{OrtAllocator}), input)
RunWithBinding(apis::OrtApi, session, run_options, binding_ptr) = ccall(Base.getproperty(apis, :RunWithBinding), OrtStatusPtr, (Ptr{OrtSession}, Ptr{OrtRunOptions}, Ptr{OrtIoBinding}), session, run_options, binding_ptr)
CreateIoBinding(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :CreateIoBinding), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{OrtIoBinding}}), session, out)
ReleaseIoBinding(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseIoBinding), Cvoid, (Ptr{OrtIoBinding}), input)
BindInput(apis::OrtApi, binding_ptr, name, val_ptr) = ccall(Base.getproperty(apis, :BindInput), OrtStatusPtr, (Ptr{OrtIoBinding}, Ptr{Cchar}, Ptr{OrtValue}), binding_ptr, name, val_ptr)
BindOutput(apis::OrtApi, binding_ptr, name, val_ptr) = ccall(Base.getproperty(apis, :BindOutput), OrtStatusPtr, (Ptr{OrtIoBinding}, Ptr{Cchar}, Ptr{OrtValue}), binding_ptr, name, val_ptr)
BindOutputToDevice(apis::OrtApi, binding_ptr, name, mem_info_ptr) = ccall(Base.getproperty(apis, :BindOutputToDevice), OrtStatusPtr, (Ptr{OrtIoBinding}, Ptr{Cchar}, Ptr{OrtMemoryInfo}), binding_ptr, name, mem_info_ptr)
GetBoundOutputNames(apis::OrtApi, binding_ptr, allocator, buffer, lengths, count) = ccall(Base.getproperty(apis, :GetBoundOutputNames), OrtStatusPtr, (Ptr{OrtIoBinding}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}, Ptr{Ptr{size_t}}, Ptr{size_t}), binding_ptr, allocator, buffer, lengths, count)
GetBoundOutputValues(apis::OrtApi, binding_ptr, allocator, output, output_count) = ccall(Base.getproperty(apis, :GetBoundOutputValues), OrtStatusPtr, (Ptr{OrtIoBinding}, Ptr{OrtAllocator}, Ptr{Ptr{Ptr{OrtValue}}}, Ptr{size_t}), binding_ptr, allocator, output, output_count)
ClearBoundInputs(apis::OrtApi, binding_ptr) = ccall(Base.getproperty(apis, :ClearBoundInputs), Cvoid, (Ptr{OrtIoBinding}), binding_ptr)
ClearBoundOutputs(apis::OrtApi, binding_ptr) = ccall(Base.getproperty(apis, :ClearBoundOutputs), Cvoid, (Ptr{OrtIoBinding}), binding_ptr)
TensorAt(apis::OrtApi, value, location_values, location_values_count, out) = ccall(Base.getproperty(apis, :TensorAt), OrtStatusPtr, (Ptr{OrtValue}, Ptr{int64_t}, size_t, Ptr{Ptr{Cvoid}}), value, location_values, location_values_count, out)
CreateAndRegisterAllocator(apis::OrtApi, env, mem_info, arena_cfg) = ccall(Base.getproperty(apis, :CreateAndRegisterAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtMemoryInfo}, Ptr{OrtArenaCfg}), env, mem_info, arena_cfg)
SetLanguageProjection(apis::OrtApi, ort_env, projection) = ccall(Base.getproperty(apis, :SetLanguageProjection), OrtStatusPtr, (Ptr{OrtEnv}, OrtLanguageProjection), ort_env, projection)
SessionGetProfilingStartTimeNs(apis::OrtApi, session, out) = ccall(Base.getproperty(apis, :SessionGetProfilingStartTimeNs), OrtStatusPtr, (Ptr{OrtSession}, Ptr{uint64_t}), session, out)
SetGlobalIntraOpNumThreads(apis::OrtApi, tp_options, intra_op_num_threads) = ccall(Base.getproperty(apis, :SetGlobalIntraOpNumThreads), OrtStatusPtr, (Ptr{OrtThreadingOptions}, Cint), tp_options, intra_op_num_threads)
SetGlobalInterOpNumThreads(apis::OrtApi, tp_options, inter_op_num_threads) = ccall(Base.getproperty(apis, :SetGlobalInterOpNumThreads), OrtStatusPtr, (Ptr{OrtThreadingOptions}, Cint), tp_options, inter_op_num_threads)
SetGlobalSpinControl(apis::OrtApi, tp_options, allow_spinning) = ccall(Base.getproperty(apis, :SetGlobalSpinControl), OrtStatusPtr, (Ptr{OrtThreadingOptions}, Cint), tp_options, allow_spinning)
AddInitializer(apis::OrtApi, options, name, val) = ccall(Base.getproperty(apis, :AddInitializer), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{OrtValue}), options, name, val)
CreateEnvWithCustomLoggerAndGlobalThreadPools(apis::OrtApi, logging_function, logger_param, log_severity_level, logid, tp_options, out) = ccall(Base.getproperty(apis, :CreateEnvWithCustomLoggerAndGlobalThreadPools), OrtStatusPtr, (OrtLoggingFunction, Ptr{Cvoid}, OrtLoggingLevel, Ptr{Cchar}, Ptr{Cvoid}, Ptr{Ptr{OrtEnv}}), logging_function, logger_param, log_severity_level, logid, tp_options, out)
SessionOptionsAppendExecutionProvider_CUDA(apis::OrtApi, options, cuda_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_CUDA), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtCUDAProviderOptions}), options, cuda_options)
SessionOptionsAppendExecutionProvider_ROCM(apis::OrtApi, options, rocm_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_ROCM), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtROCMProviderOptions}), options, rocm_options)
SessionOptionsAppendExecutionProvider_OpenVINO(apis::OrtApi, options, provider_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_OpenVINO), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtOpenVINOProviderOptions}), options, provider_options)
SetGlobalDenormalAsZero(apis::OrtApi, tp_options) = ccall(Base.getproperty(apis, :SetGlobalDenormalAsZero), OrtStatusPtr, (Ptr{OrtThreadingOptions}), tp_options)
CreateArenaCfg(apis::OrtApi, max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, out) = ccall(Base.getproperty(apis, :CreateArenaCfg), OrtStatusPtr, (size_t, Cint, Cint, Cint, Ptr{Ptr{OrtArenaCfg}}), max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, out)
ReleaseArenaCfg(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseArenaCfg), Cvoid, (Ptr{OrtArenaCfg}), input)
ModelMetadataGetGraphDescription(apis::OrtApi, model_metadata, allocator, value) = ccall(Base.getproperty(apis, :ModelMetadataGetGraphDescription), OrtStatusPtr, (Ptr{OrtModelMetadata}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), model_metadata, allocator, value)
SessionOptionsAppendExecutionProvider_TensorRT(apis::OrtApi, options, tensorrt_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_TensorRT), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtTensorRTProviderOptions}), options, tensorrt_options)
SetCurrentGpuDeviceId(apis::OrtApi, device_id) = ccall(Base.getproperty(apis, :SetCurrentGpuDeviceId), OrtStatusPtr, (Cint), device_id)
GetCurrentGpuDeviceId(apis::OrtApi, device_id) = ccall(Base.getproperty(apis, :GetCurrentGpuDeviceId), OrtStatusPtr, (Ptr{Cint}), device_id)
KernelInfoGetAttributeArray_float(apis::OrtApi, info, name, out, size) = ccall(Base.getproperty(apis, :KernelInfoGetAttributeArray_float), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{Cfloat}, Ptr{size_t}), info, name, out, size)
KernelInfoGetAttributeArray_int64(apis::OrtApi, info, name, out, size) = ccall(Base.getproperty(apis, :KernelInfoGetAttributeArray_int64), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{int64_t}, Ptr{size_t}), info, name, out, size)
CreateArenaCfgV2(apis::OrtApi, arena_config_keys, arena_config_values, num_keys, out) = ccall(Base.getproperty(apis, :CreateArenaCfgV2), OrtStatusPtr, (Ptr{Ptr{Cchar}}, Ptr{size_t}, size_t, Ptr{Ptr{OrtArenaCfg}}), arena_config_keys, arena_config_values, num_keys, out)
AddRunConfigEntry(apis::OrtApi, options, config_key, config_value) = ccall(Base.getproperty(apis, :AddRunConfigEntry), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{Cchar}, Ptr{Cchar}), options, config_key, config_value)
CreatePrepackedWeightsContainer(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreatePrepackedWeightsContainer), OrtStatusPtr, (Ptr{Ptr{OrtPrepackedWeightsContainer}}), out)
ReleasePrepackedWeightsContainer(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleasePrepackedWeightsContainer), Cvoid, (Ptr{OrtPrepackedWeightsContainer}), input)
CreateSessionWithPrepackedWeightsContainer(apis::OrtApi, env, model_path, options, prepacked_weights_container, out) = ccall(Base.getproperty(apis, :CreateSessionWithPrepackedWeightsContainer), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{wchar_t}, Ptr{OrtSessionOptions}, Ptr{OrtPrepackedWeightsContainer}, Ptr{Ptr{OrtSession}}), env, model_path, options, prepacked_weights_container, out)
CreateSessionFromArrayWithPrepackedWeightsContainer(apis::OrtApi, env, model_data, model_data_length, options, prepacked_weights_container, out) = ccall(Base.getproperty(apis, :CreateSessionFromArrayWithPrepackedWeightsContainer), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Cvoid}, size_t, Ptr{OrtSessionOptions}, Ptr{OrtPrepackedWeightsContainer}, Ptr{Ptr{OrtSession}}), env, model_data, model_data_length, options, prepacked_weights_container, out)
SessionOptionsAppendExecutionProvider_TensorRT_V2(apis::OrtApi, options, tensorrt_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_TensorRT_V2), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtTensorRTProviderOptionsV2}), options, tensorrt_options)
CreateTensorRTProviderOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateTensorRTProviderOptions), OrtStatusPtr, (Ptr{Ptr{OrtTensorRTProviderOptionsV2}}), out)
UpdateTensorRTProviderOptions(apis::OrtApi, tensorrt_options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :UpdateTensorRTProviderOptions), OrtStatusPtr, (Ptr{OrtTensorRTProviderOptionsV2}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), tensorrt_options, provider_options_keys, provider_options_values, num_keys)
GetTensorRTProviderOptionsAsString(apis::OrtApi, tensorrt_options, allocator, ptr) = ccall(Base.getproperty(apis, :GetTensorRTProviderOptionsAsString), OrtStatusPtr, (Ptr{OrtTensorRTProviderOptionsV2}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), tensorrt_options, allocator, ptr)
ReleaseTensorRTProviderOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseTensorRTProviderOptions), Cvoid, (Ptr{OrtTensorRTProviderOptionsV2}), input)
EnableOrtCustomOps(apis::OrtApi, options) = ccall(Base.getproperty(apis, :EnableOrtCustomOps), OrtStatusPtr, (Ptr{OrtSessionOptions}), options)
RegisterAllocator(apis::OrtApi, env, allocator) = ccall(Base.getproperty(apis, :RegisterAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtAllocator}), env, allocator)
UnregisterAllocator(apis::OrtApi, env, mem_info) = ccall(Base.getproperty(apis, :UnregisterAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtMemoryInfo}), env, mem_info)
IsSparseTensor(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :IsSparseTensor), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cint}), value, out)
CreateSparseTensorAsOrtValue(apis::OrtApi, allocator, dense_shape, dense_shape_len, type, out) = ccall(Base.getproperty(apis, :CreateSparseTensorAsOrtValue), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{int64_t}, size_t, ONNXTensorElementDataType, Ptr{Ptr{OrtValue}}), allocator, dense_shape, dense_shape_len, type, out)
FillSparseTensorCoo(apis::OrtApi, ort_value, data_mem_info, values_shape, values_shape_len, values, indices_data, indices_num) = ccall(Base.getproperty(apis, :FillSparseTensorCoo), OrtStatusPtr, (Ptr{OrtValue}, Ptr{OrtMemoryInfo}, Ptr{int64_t}, size_t, Ptr{Cvoid}, Ptr{int64_t}, size_t), ort_value, data_mem_info, values_shape, values_shape_len, values, indices_data, indices_num)
FillSparseTensorCsr(apis::OrtApi, ort_value, data_mem_info, values_shape, values_shape_len, values, inner_indices_data, inner_indices_num, outer_indices_data, outer_indices_num) = ccall(Base.getproperty(apis, :FillSparseTensorCsr), OrtStatusPtr, (Ptr{OrtValue}, Ptr{OrtMemoryInfo}, Ptr{int64_t}, size_t, Ptr{Cvoid}, Ptr{int64_t}, size_t, Ptr{int64_t}, size_t), ort_value, data_mem_info, values_shape, values_shape_len, values, inner_indices_data, inner_indices_num, outer_indices_data, outer_indices_num)
FillSparseTensorBlockSparse(apis::OrtApi, ort_value, data_mem_info, values_shape, values_shape_len, values, indices_shape_data, indices_shape_len, indices_data) = ccall(Base.getproperty(apis, :FillSparseTensorBlockSparse), OrtStatusPtr, (Ptr{OrtValue}, Ptr{OrtMemoryInfo}, Ptr{int64_t}, size_t, Ptr{Cvoid}, Ptr{int64_t}, size_t, Ptr{int32_t}), ort_value, data_mem_info, values_shape, values_shape_len, values, indices_shape_data, indices_shape_len, indices_data)
CreateSparseTensorWithValuesAsOrtValue(apis::OrtApi, info, p_data, dense_shape, dense_shape_len, values_shape, values_shape_len, type, out) = ccall(Base.getproperty(apis, :CreateSparseTensorWithValuesAsOrtValue), OrtStatusPtr, (Ptr{OrtMemoryInfo}, Ptr{Cvoid}, Ptr{int64_t}, size_t, Ptr{int64_t}, size_t, ONNXTensorElementDataType, Ptr{Ptr{OrtValue}}), info, p_data, dense_shape, dense_shape_len, values_shape, values_shape_len, type, out)
UseCooIndices(apis::OrtApi, ort_value, indices_data, indices_num) = ccall(Base.getproperty(apis, :UseCooIndices), OrtStatusPtr, (Ptr{OrtValue}, Ptr{int64_t}, size_t), ort_value, indices_data, indices_num)
UseCsrIndices(apis::OrtApi, ort_value, inner_data, inner_num, outer_data, outer_num) = ccall(Base.getproperty(apis, :UseCsrIndices), OrtStatusPtr, (Ptr{OrtValue}, Ptr{int64_t}, size_t, Ptr{int64_t}, size_t), ort_value, inner_data, inner_num, outer_data, outer_num)
UseBlockSparseIndices(apis::OrtApi, ort_value, indices_shape, indices_shape_len, indices_data) = ccall(Base.getproperty(apis, :UseBlockSparseIndices), OrtStatusPtr, (Ptr{OrtValue}, Ptr{int64_t}, size_t, Ptr{int32_t}), ort_value, indices_shape, indices_shape_len, indices_data)
GetSparseTensorFormat(apis::OrtApi, ort_value, out) = ccall(Base.getproperty(apis, :GetSparseTensorFormat), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cvoid}), ort_value, out)
GetSparseTensorValuesTypeAndShape(apis::OrtApi, ort_value, out) = ccall(Base.getproperty(apis, :GetSparseTensorValuesTypeAndShape), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), ort_value, out)
GetSparseTensorValues(apis::OrtApi, ort_value, out) = ccall(Base.getproperty(apis, :GetSparseTensorValues), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{Cvoid}}), ort_value, out)
GetSparseTensorIndicesTypeShape(apis::OrtApi, ort_value, indices_format, out) = ccall(Base.getproperty(apis, :GetSparseTensorIndicesTypeShape), OrtStatusPtr, (Ptr{OrtValue}, Cvoid, Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), ort_value, indices_format, out)
GetSparseTensorIndices(apis::OrtApi, ort_value, indices_format, num_indices, indices) = ccall(Base.getproperty(apis, :GetSparseTensorIndices), OrtStatusPtr, (Ptr{OrtValue}, Cvoid, Ptr{size_t}, Ptr{Ptr{Cvoid}}), ort_value, indices_format, num_indices, indices)
HasValue(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :HasValue), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Cint}), value, out)
KernelContext_GetGPUComputeStream(apis::OrtApi, context, out) = ccall(Base.getproperty(apis, :KernelContext_GetGPUComputeStream), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{Ptr{Cvoid}}), context, out)
GetTensorMemoryInfo(apis::OrtApi, value, mem_info) = ccall(Base.getproperty(apis, :GetTensorMemoryInfo), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{OrtMemoryInfo}}), value, mem_info)
GetExecutionProviderApi(apis::OrtApi, provider_name, version, provider_api) = ccall(Base.getproperty(apis, :GetExecutionProviderApi), OrtStatusPtr, (Ptr{Cchar}, uint32_t, Ptr{Ptr{Cvoid}}), provider_name, version, provider_api)
SessionOptionsSetCustomCreateThreadFn(apis::OrtApi, options, ort_custom_create_thread_fn) = ccall(Base.getproperty(apis, :SessionOptionsSetCustomCreateThreadFn), OrtStatusPtr, (Ptr{OrtSessionOptions}, OrtCustomCreateThreadFn), options, ort_custom_create_thread_fn)
SessionOptionsSetCustomThreadCreationOptions(apis::OrtApi, options, ort_custom_thread_creation_options) = ccall(Base.getproperty(apis, :SessionOptionsSetCustomThreadCreationOptions), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cvoid}), options, ort_custom_thread_creation_options)
SessionOptionsSetCustomJoinThreadFn(apis::OrtApi, options, ort_custom_join_thread_fn) = ccall(Base.getproperty(apis, :SessionOptionsSetCustomJoinThreadFn), OrtStatusPtr, (Ptr{OrtSessionOptions}, OrtCustomJoinThreadFn), options, ort_custom_join_thread_fn)
SetGlobalCustomCreateThreadFn(apis::OrtApi, tp_options, ort_custom_create_thread_fn) = ccall(Base.getproperty(apis, :SetGlobalCustomCreateThreadFn), OrtStatusPtr, (Ptr{OrtThreadingOptions}, OrtCustomCreateThreadFn), tp_options, ort_custom_create_thread_fn)
SetGlobalCustomThreadCreationOptions(apis::OrtApi, tp_options, ort_custom_thread_creation_options) = ccall(Base.getproperty(apis, :SetGlobalCustomThreadCreationOptions), OrtStatusPtr, (Ptr{OrtThreadingOptions}, Ptr{Cvoid}), tp_options, ort_custom_thread_creation_options)
SetGlobalCustomJoinThreadFn(apis::OrtApi, tp_options, ort_custom_join_thread_fn) = ccall(Base.getproperty(apis, :SetGlobalCustomJoinThreadFn), OrtStatusPtr, (Ptr{OrtThreadingOptions}, OrtCustomJoinThreadFn), tp_options, ort_custom_join_thread_fn)
SynchronizeBoundInputs(apis::OrtApi, binding_ptr) = ccall(Base.getproperty(apis, :SynchronizeBoundInputs), OrtStatusPtr, (Ptr{OrtIoBinding}), binding_ptr)
SynchronizeBoundOutputs(apis::OrtApi, binding_ptr) = ccall(Base.getproperty(apis, :SynchronizeBoundOutputs), OrtStatusPtr, (Ptr{OrtIoBinding}), binding_ptr)
SessionOptionsAppendExecutionProvider_CUDA_V2(apis::OrtApi, options, cuda_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_CUDA_V2), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtCUDAProviderOptionsV2}), options, cuda_options)
CreateCUDAProviderOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateCUDAProviderOptions), OrtStatusPtr, (Ptr{Ptr{OrtCUDAProviderOptionsV2}}), out)
UpdateCUDAProviderOptions(apis::OrtApi, cuda_options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :UpdateCUDAProviderOptions), OrtStatusPtr, (Ptr{OrtCUDAProviderOptionsV2}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), cuda_options, provider_options_keys, provider_options_values, num_keys)
GetCUDAProviderOptionsAsString(apis::OrtApi, cuda_options, allocator, ptr) = ccall(Base.getproperty(apis, :GetCUDAProviderOptionsAsString), OrtStatusPtr, (Ptr{OrtCUDAProviderOptionsV2}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), cuda_options, allocator, ptr)
ReleaseCUDAProviderOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseCUDAProviderOptions), Cvoid, (Ptr{OrtCUDAProviderOptionsV2}), input)
SessionOptionsAppendExecutionProvider_MIGraphX(apis::OrtApi, options, migraphx_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_MIGraphX), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtMIGraphXProviderOptions}), options, migraphx_options)
AddExternalInitializers(apis::OrtApi, options, initializer_names, initializers, num_initializers) = ccall(Base.getproperty(apis, :AddExternalInitializers), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{OrtValue}}, size_t), options, initializer_names, initializers, num_initializers)
CreateOpAttr(apis::OrtApi, name, data, len, type, op_attr) = ccall(Base.getproperty(apis, :CreateOpAttr), OrtStatusPtr, (Ptr{Cchar}, Ptr{Cvoid}, Cint, OrtOpAttrType, Ptr{Ptr{OrtOpAttr}}), name, data, len, type, op_attr)
ReleaseOpAttr(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseOpAttr), Cvoid, (Ptr{OrtOpAttr}), input)
CreateOp(apis::OrtApi, info, op_name, domain, version, type_constraint_names, type_constraint_values, type_constraint_count, attr_values, attr_count, input_count, output_count, ort_op) = ccall(Base.getproperty(apis, :CreateOp), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{Cchar}, Cint, Ptr{Ptr{Cchar}}, Ptr{ONNXTensorElementDataType}, Cint, Ptr{Ptr{OrtOpAttr}}, Cint, Cint, Cint, Ptr{Ptr{OrtOp}}), info, op_name, domain, version, type_constraint_names, type_constraint_values, type_constraint_count, attr_values, attr_count, input_count, output_count, ort_op)
InvokeOp(apis::OrtApi, context, ort_op, input_values, input_count, output_values, output_count) = ccall(Base.getproperty(apis, :InvokeOp), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{OrtOp}, Ptr{Ptr{OrtValue}}, Cint, Ptr{Ptr{OrtValue}}, Cint), context, ort_op, input_values, input_count, output_values, output_count)
ReleaseOp(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseOp), Cvoid, (Ptr{OrtOp}), input)
SessionOptionsAppendExecutionProvider(apis::OrtApi, options, provider_name, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), options, provider_name, provider_options_keys, provider_options_values, num_keys)
CopyKernelInfo(apis::OrtApi, info, info_copy) = ccall(Base.getproperty(apis, :CopyKernelInfo), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Ptr{OrtKernelInfo}}), info, info_copy)
ReleaseKernelInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseKernelInfo), Cvoid, (Ptr{OrtKernelInfo}), input)
GetTrainingApi(apis::OrtApi, version) = ccall(Base.getproperty(apis, :GetTrainingApi), Ptr{OrtTrainingApi}, (uint32_t), version)
SessionOptionsAppendExecutionProvider_CANN(apis::OrtApi, options, cann_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_CANN), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtCANNProviderOptions}), options, cann_options)
CreateCANNProviderOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateCANNProviderOptions), OrtStatusPtr, (Ptr{Ptr{OrtCANNProviderOptions}}), out)
UpdateCANNProviderOptions(apis::OrtApi, cann_options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :UpdateCANNProviderOptions), OrtStatusPtr, (Ptr{OrtCANNProviderOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), cann_options, provider_options_keys, provider_options_values, num_keys)
GetCANNProviderOptionsAsString(apis::OrtApi, cann_options, allocator, ptr) = ccall(Base.getproperty(apis, :GetCANNProviderOptionsAsString), OrtStatusPtr, (Ptr{OrtCANNProviderOptions}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), cann_options, allocator, ptr)
ReleaseCANNProviderOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseCANNProviderOptions), Cvoid, (Ptr{OrtCANNProviderOptions}), input)
MemoryInfoGetDeviceType(apis::OrtApi, ptr, out) = ccall(Base.getproperty(apis, :MemoryInfoGetDeviceType), Cvoid, (Ptr{OrtMemoryInfo}, Ptr{OrtMemoryInfoDeviceType}), ptr, out)
UpdateEnvWithCustomLogLevel(apis::OrtApi, ort_env, log_severity_level) = ccall(Base.getproperty(apis, :UpdateEnvWithCustomLogLevel), OrtStatusPtr, (Ptr{OrtEnv}, OrtLoggingLevel), ort_env, log_severity_level)
SetGlobalIntraOpThreadAffinity(apis::OrtApi, tp_options, affinity_string) = ccall(Base.getproperty(apis, :SetGlobalIntraOpThreadAffinity), OrtStatusPtr, (Ptr{OrtThreadingOptions}, Ptr{Cchar}), tp_options, affinity_string)
RegisterCustomOpsLibrary_V2(apis::OrtApi, options, library_name) = ccall(Base.getproperty(apis, :RegisterCustomOpsLibrary_V2), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{wchar_t}), options, library_name)
RegisterCustomOpsUsingFunction(apis::OrtApi, options, registration_func_name) = ccall(Base.getproperty(apis, :RegisterCustomOpsUsingFunction), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}), options, registration_func_name)
KernelInfo_GetInputCount(apis::OrtApi, info, out) = ccall(Base.getproperty(apis, :KernelInfo_GetInputCount), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{size_t}), info, out)
KernelInfo_GetOutputCount(apis::OrtApi, info, out) = ccall(Base.getproperty(apis, :KernelInfo_GetOutputCount), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{size_t}), info, out)
KernelInfo_GetInputName(apis::OrtApi, info, index, out, size) = ccall(Base.getproperty(apis, :KernelInfo_GetInputName), OrtStatusPtr, (Ptr{OrtKernelInfo}, size_t, Ptr{Cchar}, Ptr{size_t}), info, index, out, size)
KernelInfo_GetOutputName(apis::OrtApi, info, index, out, size) = ccall(Base.getproperty(apis, :KernelInfo_GetOutputName), OrtStatusPtr, (Ptr{OrtKernelInfo}, size_t, Ptr{Cchar}, Ptr{size_t}), info, index, out, size)
KernelInfo_GetInputTypeInfo(apis::OrtApi, info, index, type_info) = ccall(Base.getproperty(apis, :KernelInfo_GetInputTypeInfo), OrtStatusPtr, (Ptr{OrtKernelInfo}, size_t, Ptr{Ptr{OrtTypeInfo}}), info, index, type_info)
KernelInfo_GetOutputTypeInfo(apis::OrtApi, info, index, type_info) = ccall(Base.getproperty(apis, :KernelInfo_GetOutputTypeInfo), OrtStatusPtr, (Ptr{OrtKernelInfo}, size_t, Ptr{Ptr{OrtTypeInfo}}), info, index, type_info)
KernelInfoGetAttribute_tensor(apis::OrtApi, info, name, allocator, out) = ccall(Base.getproperty(apis, :KernelInfoGetAttribute_tensor), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{OrtAllocator}, Ptr{Ptr{OrtValue}}), info, name, allocator, out)
HasSessionConfigEntry(apis::OrtApi, options, config_key, out) = ccall(Base.getproperty(apis, :HasSessionConfigEntry), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{Cint}), options, config_key, out)
GetSessionConfigEntry(apis::OrtApi, options, config_key, config_value, size) = ccall(Base.getproperty(apis, :GetSessionConfigEntry), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Cchar}, Ptr{Cchar}, Ptr{size_t}), options, config_key, config_value, size)
SessionOptionsAppendExecutionProvider_Dnnl(apis::OrtApi, options, dnnl_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_Dnnl), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtDnnlProviderOptions}), options, dnnl_options)
CreateDnnlProviderOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateDnnlProviderOptions), OrtStatusPtr, (Ptr{Ptr{OrtDnnlProviderOptions}}), out)
UpdateDnnlProviderOptions(apis::OrtApi, dnnl_options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :UpdateDnnlProviderOptions), OrtStatusPtr, (Ptr{OrtDnnlProviderOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), dnnl_options, provider_options_keys, provider_options_values, num_keys)
GetDnnlProviderOptionsAsString(apis::OrtApi, dnnl_options, allocator, ptr) = ccall(Base.getproperty(apis, :GetDnnlProviderOptionsAsString), OrtStatusPtr, (Ptr{OrtDnnlProviderOptions}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), dnnl_options, allocator, ptr)
ReleaseDnnlProviderOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseDnnlProviderOptions), Cvoid, (Ptr{OrtDnnlProviderOptions}), input)
KernelInfo_GetNodeName(apis::OrtApi, info, out, size) = ccall(Base.getproperty(apis, :KernelInfo_GetNodeName), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Cchar}, Ptr{size_t}), info, out, size)
KernelInfo_GetLogger(apis::OrtApi, info, logger) = ccall(Base.getproperty(apis, :KernelInfo_GetLogger), OrtStatusPtr, (Ptr{OrtKernelInfo}, Ptr{Ptr{OrtLogger}}), info, logger)
KernelContext_GetLogger(apis::OrtApi, context, logger) = ccall(Base.getproperty(apis, :KernelContext_GetLogger), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{Ptr{OrtLogger}}), context, logger)
Logger_LogMessage(apis::OrtApi, logger, log_severity_level, message, file_path, line_number, func_name) = ccall(Base.getproperty(apis, :Logger_LogMessage), OrtStatusPtr, (Ptr{OrtLogger}, OrtLoggingLevel, Ptr{Cchar}, Ptr{wchar_t}, Cint, Ptr{Cchar}), logger, log_severity_level, message, file_path, line_number, func_name)
Logger_GetLoggingSeverityLevel(apis::OrtApi, logger, out) = ccall(Base.getproperty(apis, :Logger_GetLoggingSeverityLevel), OrtStatusPtr, (Ptr{OrtLogger}, Ptr{OrtLoggingLevel}), logger, out)
KernelInfoGetConstantInput_tensor(apis::OrtApi, info, index, is_constant, out) = ccall(Base.getproperty(apis, :KernelInfoGetConstantInput_tensor), OrtStatusPtr, (Ptr{OrtKernelInfo}, size_t, Ptr{Cint}, Ptr{Ptr{OrtValue}}), info, index, is_constant, out)
CastTypeInfoToOptionalTypeInfo(apis::OrtApi, type_info, out) = ccall(Base.getproperty(apis, :CastTypeInfoToOptionalTypeInfo), OrtStatusPtr, (Ptr{OrtTypeInfo}, Ptr{Ptr{OrtOptionalTypeInfo}}), type_info, out)
GetOptionalContainedTypeInfo(apis::OrtApi, optional_type_info, out) = ccall(Base.getproperty(apis, :GetOptionalContainedTypeInfo), OrtStatusPtr, (Ptr{OrtOptionalTypeInfo}, Ptr{Ptr{OrtTypeInfo}}), optional_type_info, out)
GetResizedStringTensorElementBuffer(apis::OrtApi, value, index, length_in_bytes, buffer) = ccall(Base.getproperty(apis, :GetResizedStringTensorElementBuffer), OrtStatusPtr, (Ptr{OrtValue}, size_t, size_t, Ptr{Ptr{Cchar}}), value, index, length_in_bytes, buffer)
KernelContext_GetAllocator(apis::OrtApi, context, mem_info, out) = ccall(Base.getproperty(apis, :KernelContext_GetAllocator), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{OrtMemoryInfo}, Ptr{Ptr{OrtAllocator}}), context, mem_info, out)
GetBuildInfoString(apis::OrtApi) = ccall(Base.getproperty(apis, :GetBuildInfoString), Ptr{Cchar}, (), )
CreateROCMProviderOptions(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateROCMProviderOptions), OrtStatusPtr, (Ptr{Ptr{OrtROCMProviderOptions}}), out)
UpdateROCMProviderOptions(apis::OrtApi, rocm_options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :UpdateROCMProviderOptions), OrtStatusPtr, (Ptr{OrtROCMProviderOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), rocm_options, provider_options_keys, provider_options_values, num_keys)
GetROCMProviderOptionsAsString(apis::OrtApi, rocm_options, allocator, ptr) = ccall(Base.getproperty(apis, :GetROCMProviderOptionsAsString), OrtStatusPtr, (Ptr{OrtROCMProviderOptions}, Ptr{OrtAllocator}, Ptr{Ptr{Cchar}}), rocm_options, allocator, ptr)
ReleaseROCMProviderOptions(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseROCMProviderOptions), Cvoid, (Ptr{OrtROCMProviderOptions}), input)
CreateAndRegisterAllocatorV2(apis::OrtApi, env, provider_type, mem_info, arena_cfg, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :CreateAndRegisterAllocatorV2), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Cchar}, Ptr{OrtMemoryInfo}, Ptr{OrtArenaCfg}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), env, provider_type, mem_info, arena_cfg, provider_options_keys, provider_options_values, num_keys)
RunAsync(apis::OrtApi, session, run_options, input_names, input, input_len, output_names, output_names_len, output, run_async_callback, user_data) = ccall(Base.getproperty(apis, :RunAsync), OrtStatusPtr, (Ptr{OrtSession}, Ptr{OrtRunOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{OrtValue}}, size_t, Ptr{Ptr{Cchar}}, size_t, Ptr{Ptr{OrtValue}}, RunAsyncCallbackFn, Ptr{Cvoid}), session, run_options, input_names, input, input_len, output_names, output_names_len, output, run_async_callback, user_data)
UpdateTensorRTProviderOptionsWithValue(apis::OrtApi, tensorrt_options, key, value) = ccall(Base.getproperty(apis, :UpdateTensorRTProviderOptionsWithValue), OrtStatusPtr, (Ptr{OrtTensorRTProviderOptionsV2}, Ptr{Cchar}, Ptr{Cvoid}), tensorrt_options, key, value)
GetTensorRTProviderOptionsByName(apis::OrtApi, tensorrt_options, key, ptr) = ccall(Base.getproperty(apis, :GetTensorRTProviderOptionsByName), OrtStatusPtr, (Ptr{OrtTensorRTProviderOptionsV2}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}), tensorrt_options, key, ptr)
UpdateCUDAProviderOptionsWithValue(apis::OrtApi, cuda_options, key, value) = ccall(Base.getproperty(apis, :UpdateCUDAProviderOptionsWithValue), OrtStatusPtr, (Ptr{OrtCUDAProviderOptionsV2}, Ptr{Cchar}, Ptr{Cvoid}), cuda_options, key, value)
GetCUDAProviderOptionsByName(apis::OrtApi, cuda_options, key, ptr) = ccall(Base.getproperty(apis, :GetCUDAProviderOptionsByName), OrtStatusPtr, (Ptr{OrtCUDAProviderOptionsV2}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}), cuda_options, key, ptr)
KernelContext_GetResource(apis::OrtApi, context, resource_version, resource_id, resource) = ccall(Base.getproperty(apis, :KernelContext_GetResource), OrtStatusPtr, (Ptr{OrtKernelContext}, Cint, Cint, Ptr{Ptr{Cvoid}}), context, resource_version, resource_id, resource)
SetUserLoggingFunction(apis::OrtApi, options, user_logging_function, user_logging_param) = ccall(Base.getproperty(apis, :SetUserLoggingFunction), OrtStatusPtr, (Ptr{OrtSessionOptions}, OrtLoggingFunction, Ptr{Cvoid}), options, user_logging_function, user_logging_param)
ShapeInferContext_GetInputCount(apis::OrtApi, context, out) = ccall(Base.getproperty(apis, :ShapeInferContext_GetInputCount), OrtStatusPtr, (Ptr{OrtShapeInferContext}, Ptr{size_t}), context, out)
ShapeInferContext_GetInputTypeShape(apis::OrtApi, context, index, info) = ccall(Base.getproperty(apis, :ShapeInferContext_GetInputTypeShape), OrtStatusPtr, (Ptr{OrtShapeInferContext}, size_t, Ptr{Ptr{OrtTensorTypeAndShapeInfo}}), context, index, info)
ShapeInferContext_GetAttribute(apis::OrtApi, context, attr_name, attr) = ccall(Base.getproperty(apis, :ShapeInferContext_GetAttribute), OrtStatusPtr, (Ptr{OrtShapeInferContext}, Ptr{Cchar}, Ptr{Ptr{OrtOpAttr}}), context, attr_name, attr)
ShapeInferContext_SetOutputTypeShape(apis::OrtApi, context, index, info) = ccall(Base.getproperty(apis, :ShapeInferContext_SetOutputTypeShape), OrtStatusPtr, (Ptr{OrtShapeInferContext}, size_t, Ptr{OrtTensorTypeAndShapeInfo}), context, index, info)
SetSymbolicDimensions(apis::OrtApi, info, dim_params, dim_params_length) = ccall(Base.getproperty(apis, :SetSymbolicDimensions), OrtStatusPtr, (Ptr{OrtTensorTypeAndShapeInfo}, Ptr{Ptr{Cchar}}, size_t), info, dim_params, dim_params_length)
ReadOpAttr(apis::OrtApi, op_attr, type, data, len, out) = ccall(Base.getproperty(apis, :ReadOpAttr), OrtStatusPtr, (Ptr{OrtOpAttr}, OrtOpAttrType, Ptr{Cvoid}, size_t, Ptr{size_t}), op_attr, type, data, len, out)
SetDeterministicCompute(apis::OrtApi, options, value) = ccall(Base.getproperty(apis, :SetDeterministicCompute), OrtStatusPtr, (Ptr{OrtSessionOptions}, Bool), options, value)
KernelContext_ParallelFor(apis::OrtApi, context, fn, total, num_batch, usr_data) = ccall(Base.getproperty(apis, :KernelContext_ParallelFor), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{Cvoid}, size_t, size_t, Ptr{Cvoid}), context, fn, total, num_batch, usr_data)
SessionOptionsAppendExecutionProvider_OpenVINO_V2(apis::OrtApi, options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_OpenVINO_V2), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), options, provider_options_keys, provider_options_values, num_keys)
SessionOptionsAppendExecutionProvider_VitisAI(apis::OrtApi, options, provider_options_keys, provider_options_values, num_keys) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_VitisAI), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), options, provider_options_keys, provider_options_values, num_keys)
KernelContext_GetScratchBuffer(apis::OrtApi, context, mem_info, count_or_bytes, out) = ccall(Base.getproperty(apis, :KernelContext_GetScratchBuffer), OrtStatusPtr, (Ptr{OrtKernelContext}, Ptr{OrtMemoryInfo}, size_t, Ptr{Ptr{Cvoid}}), context, mem_info, count_or_bytes, out)
KernelInfoGetAllocator(apis::OrtApi, info, mem_type, out) = ccall(Base.getproperty(apis, :KernelInfoGetAllocator), OrtStatusPtr, (Ptr{OrtKernelInfo}, OrtMemType, Ptr{Ptr{OrtAllocator}}), info, mem_type, out)
AddExternalInitializersFromFilesInMemory(apis::OrtApi, options, external_initializer_file_names, external_initializer_file_buffer_array, external_initializer_file_lengths, num_external_initializer_files) = ccall(Base.getproperty(apis, :AddExternalInitializersFromFilesInMemory), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{wchar_t}}, Ptr{Ptr{Cchar}}, Ptr{size_t}, size_t), options, external_initializer_file_names, external_initializer_file_buffer_array, external_initializer_file_lengths, num_external_initializer_files)
CreateLoraAdapter(apis::OrtApi, adapter_file_path, allocator, out) = ccall(Base.getproperty(apis, :CreateLoraAdapter), OrtStatusPtr, (Ptr{wchar_t}, Ptr{OrtAllocator}, Ptr{Ptr{OrtLoraAdapter}}), adapter_file_path, allocator, out)
CreateLoraAdapterFromArray(apis::OrtApi, bytes, num_bytes, allocator, out) = ccall(Base.getproperty(apis, :CreateLoraAdapterFromArray), OrtStatusPtr, (Ptr{Cvoid}, size_t, Ptr{OrtAllocator}, Ptr{Ptr{OrtLoraAdapter}}), bytes, num_bytes, allocator, out)
ReleaseLoraAdapter(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseLoraAdapter), Cvoid, (Ptr{OrtLoraAdapter}), input)
RunOptionsAddActiveLoraAdapter(apis::OrtApi, options, adapter) = ccall(Base.getproperty(apis, :RunOptionsAddActiveLoraAdapter), OrtStatusPtr, (Ptr{OrtRunOptions}, Ptr{OrtLoraAdapter}), options, adapter)
SetEpDynamicOptions(apis::OrtApi, sess, keys, values, kv_len) = ccall(Base.getproperty(apis, :SetEpDynamicOptions), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), sess, keys, values, kv_len)
ReleaseValueInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseValueInfo), Cvoid, (Ptr{OrtValueInfo}), input)
ReleaseNode(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseNode), Cvoid, (Ptr{OrtNode}), input)
ReleaseGraph(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseGraph), Cvoid, (Ptr{OrtGraph}), input)
ReleaseModel(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseModel), Cvoid, (Ptr{OrtModel}), input)
GetValueInfoName(apis::OrtApi, value_info, name) = ccall(Base.getproperty(apis, :GetValueInfoName), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{Cchar}}), value_info, name)
GetValueInfoTypeInfo(apis::OrtApi, value_info, type_info) = ccall(Base.getproperty(apis, :GetValueInfoTypeInfo), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{OrtTypeInfo}}), value_info, type_info)
GetModelEditorApi(apis::OrtApi) = ccall(Base.getproperty(apis, :GetModelEditorApi), Ptr{OrtModelEditorApi}, (), )
CreateTensorWithDataAndDeleterAsOrtValue(apis::OrtApi, deleter, p_data, p_data_len, shape, shape_len, type, out) = ccall(Base.getproperty(apis, :CreateTensorWithDataAndDeleterAsOrtValue), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{Cvoid}, size_t, Ptr{int64_t}, size_t, ONNXTensorElementDataType, Ptr{Ptr{OrtValue}}), deleter, p_data, p_data_len, shape, shape_len, type, out)
SessionOptionsSetLoadCancellationFlag(apis::OrtApi, options, cancel) = ccall(Base.getproperty(apis, :SessionOptionsSetLoadCancellationFlag), OrtStatusPtr, (Ptr{OrtSessionOptions}, Bool), options, cancel)
GetCompileApi(apis::OrtApi) = ccall(Base.getproperty(apis, :GetCompileApi), Ptr{OrtCompileApi}, (), )
CreateKeyValuePairs(apis::OrtApi, out) = ccall(Base.getproperty(apis, :CreateKeyValuePairs), Cvoid, (Ptr{Ptr{OrtKeyValuePairs}}), out)
AddKeyValuePair(apis::OrtApi, kvps, key, value) = ccall(Base.getproperty(apis, :AddKeyValuePair), Cvoid, (Ptr{OrtKeyValuePairs}, Ptr{Cchar}, Ptr{Cchar}), kvps, key, value)
GetKeyValue(apis::OrtApi, kvps, key) = ccall(Base.getproperty(apis, :GetKeyValue), Ptr{Cchar}, (Ptr{OrtKeyValuePairs}, Ptr{Cchar}), kvps, key)
GetKeyValuePairs(apis::OrtApi, kvps, keys, values, num_entries) = ccall(Base.getproperty(apis, :GetKeyValuePairs), Cvoid, (Ptr{OrtKeyValuePairs}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{size_t}), kvps, keys, values, num_entries)
RemoveKeyValuePair(apis::OrtApi, kvps, key) = ccall(Base.getproperty(apis, :RemoveKeyValuePair), Cvoid, (Ptr{OrtKeyValuePairs}, Ptr{Cchar}), kvps, key)
ReleaseKeyValuePairs(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseKeyValuePairs), Cvoid, (Ptr{OrtKeyValuePairs}), input)
RegisterExecutionProviderLibrary(apis::OrtApi, env, registration_name, path) = ccall(Base.getproperty(apis, :RegisterExecutionProviderLibrary), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Cchar}, Ptr{wchar_t}), env, registration_name, path)
UnregisterExecutionProviderLibrary(apis::OrtApi, env, registration_name) = ccall(Base.getproperty(apis, :UnregisterExecutionProviderLibrary), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Cchar}), env, registration_name)
GetEpDevices(apis::OrtApi, env, ep_devices, num_ep_devices) = ccall(Base.getproperty(apis, :GetEpDevices), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Ptr{Ptr{OrtEpDevice}}}, Ptr{size_t}), env, ep_devices, num_ep_devices)
SessionOptionsAppendExecutionProvider_V2(apis::OrtApi, session_options, env, ep_devices, num_ep_devices, ep_option_keys, ep_option_vals, num_ep_options) = ccall(Base.getproperty(apis, :SessionOptionsAppendExecutionProvider_V2), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{OrtEnv}, Ptr{Ptr{OrtEpDevice}}, size_t, Ptr{Ptr{Cchar}}, Ptr{Ptr{Cchar}}, size_t), session_options, env, ep_devices, num_ep_devices, ep_option_keys, ep_option_vals, num_ep_options)
SessionOptionsSetEpSelectionPolicy(apis::OrtApi, session_options, policy) = ccall(Base.getproperty(apis, :SessionOptionsSetEpSelectionPolicy), OrtStatusPtr, (Ptr{OrtSessionOptions}, OrtExecutionProviderDevicePolicy), session_options, policy)
SessionOptionsSetEpSelectionPolicyDelegate(apis::OrtApi, session_options, delegate, delegate_state) = ccall(Base.getproperty(apis, :SessionOptionsSetEpSelectionPolicyDelegate), OrtStatusPtr, (Ptr{OrtSessionOptions}, EpSelectionDelegate, Ptr{Cvoid}), session_options, delegate, delegate_state)
HardwareDevice_Type(apis::OrtApi, device) = ccall(Base.getproperty(apis, :HardwareDevice_Type), OrtHardwareDeviceType, (Ptr{OrtHardwareDevice}), device)
HardwareDevice_VendorId(apis::OrtApi, device) = ccall(Base.getproperty(apis, :HardwareDevice_VendorId), uint32_t, (Ptr{OrtHardwareDevice}), device)
HardwareDevice_Vendor(apis::OrtApi, device) = ccall(Base.getproperty(apis, :HardwareDevice_Vendor), Ptr{Cchar}, (Ptr{OrtHardwareDevice}), device)
HardwareDevice_DeviceId(apis::OrtApi, device) = ccall(Base.getproperty(apis, :HardwareDevice_DeviceId), uint32_t, (Ptr{OrtHardwareDevice}), device)
HardwareDevice_Metadata(apis::OrtApi, device) = ccall(Base.getproperty(apis, :HardwareDevice_Metadata), Ptr{OrtKeyValuePairs}, (Ptr{OrtHardwareDevice}), device)
EpDevice_EpName(apis::OrtApi, ep_device) = ccall(Base.getproperty(apis, :EpDevice_EpName), Ptr{Cchar}, (Ptr{OrtEpDevice}), ep_device)
EpDevice_EpVendor(apis::OrtApi, ep_device) = ccall(Base.getproperty(apis, :EpDevice_EpVendor), Ptr{Cchar}, (Ptr{OrtEpDevice}), ep_device)
EpDevice_EpMetadata(apis::OrtApi, ep_device) = ccall(Base.getproperty(apis, :EpDevice_EpMetadata), Ptr{OrtKeyValuePairs}, (Ptr{OrtEpDevice}), ep_device)
EpDevice_EpOptions(apis::OrtApi, ep_device) = ccall(Base.getproperty(apis, :EpDevice_EpOptions), Ptr{OrtKeyValuePairs}, (Ptr{OrtEpDevice}), ep_device)
EpDevice_Device(apis::OrtApi, ep_device) = ccall(Base.getproperty(apis, :EpDevice_Device), Ptr{OrtHardwareDevice}, (Ptr{OrtEpDevice}), ep_device)
GetEpApi(apis::OrtApi) = ccall(Base.getproperty(apis, :GetEpApi), Ptr{OrtEpApi}, (), )
GetTensorSizeInBytes(apis::OrtApi, ort_value, size) = ccall(Base.getproperty(apis, :GetTensorSizeInBytes), OrtStatusPtr, (Ptr{OrtValue}, Ptr{size_t}), ort_value, size)
AllocatorGetStats(apis::OrtApi, ort_allocator, out) = ccall(Base.getproperty(apis, :AllocatorGetStats), OrtStatusPtr, (Ptr{OrtAllocator}, Ptr{Ptr{OrtKeyValuePairs}}), ort_allocator, out)
CreateMemoryInfo_V2(apis::OrtApi, name, device_type, vendor_id, device_id, mem_type, alignment, allocator_type, out) = ccall(Base.getproperty(apis, :CreateMemoryInfo_V2), OrtStatusPtr, (Ptr{Cchar}, Cvoid, uint32_t, int32_t, Cvoid, size_t, Cvoid, Ptr{Ptr{OrtMemoryInfo}}), name, device_type, vendor_id, device_id, mem_type, alignment, allocator_type, out)
MemoryInfoGetDeviceMemType(apis::OrtApi, ptr) = ccall(Base.getproperty(apis, :MemoryInfoGetDeviceMemType), OrtDeviceMemoryType, (Ptr{OrtMemoryInfo}), ptr)
MemoryInfoGetVendorId(apis::OrtApi, ptr) = ccall(Base.getproperty(apis, :MemoryInfoGetVendorId), uint32_t, (Ptr{OrtMemoryInfo}), ptr)
ValueInfo_GetValueProducer(apis::OrtApi, value_info, producer_node, producer_output_index) = ccall(Base.getproperty(apis, :ValueInfo_GetValueProducer), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{OrtNode}}, Ptr{size_t}), value_info, producer_node, producer_output_index)
ValueInfo_GetValueNumConsumers(apis::OrtApi, value_info, num_consumers) = ccall(Base.getproperty(apis, :ValueInfo_GetValueNumConsumers), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{size_t}), value_info, num_consumers)
ValueInfo_GetValueConsumers(apis::OrtApi, value_info, nodes, input_indices, num_consumers) = ccall(Base.getproperty(apis, :ValueInfo_GetValueConsumers), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{OrtNode}}, Ptr{int64_t}, size_t), value_info, nodes, input_indices, num_consumers)
ValueInfo_GetInitializerValue(apis::OrtApi, value_info, initializer_value) = ccall(Base.getproperty(apis, :ValueInfo_GetInitializerValue), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{OrtValue}}), value_info, initializer_value)
ValueInfo_GetExternalInitializerInfo(apis::OrtApi, value_info, info) = ccall(Base.getproperty(apis, :ValueInfo_GetExternalInitializerInfo), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Ptr{OrtExternalInitializerInfo}}), value_info, info)
ValueInfo_IsRequiredGraphInput(apis::OrtApi, value_info, is_required_graph_input) = ccall(Base.getproperty(apis, :ValueInfo_IsRequiredGraphInput), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Bool}), value_info, is_required_graph_input)
ValueInfo_IsOptionalGraphInput(apis::OrtApi, value_info, is_optional_graph_input) = ccall(Base.getproperty(apis, :ValueInfo_IsOptionalGraphInput), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Bool}), value_info, is_optional_graph_input)
ValueInfo_IsGraphOutput(apis::OrtApi, value_info, is_graph_output) = ccall(Base.getproperty(apis, :ValueInfo_IsGraphOutput), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Bool}), value_info, is_graph_output)
ValueInfo_IsConstantInitializer(apis::OrtApi, value_info, is_constant_initializer) = ccall(Base.getproperty(apis, :ValueInfo_IsConstantInitializer), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Bool}), value_info, is_constant_initializer)
ValueInfo_IsFromOuterScope(apis::OrtApi, value_info, is_from_outer_scope) = ccall(Base.getproperty(apis, :ValueInfo_IsFromOuterScope), OrtStatusPtr, (Ptr{OrtValueInfo}, Ptr{Bool}), value_info, is_from_outer_scope)
Graph_GetName(apis::OrtApi, graph, graph_name) = ccall(Base.getproperty(apis, :Graph_GetName), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{Cchar}}), graph, graph_name)
Graph_GetModelPath(apis::OrtApi, graph, model_path) = ccall(Base.getproperty(apis, :Graph_GetModelPath), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{wchar_t}}), graph, model_path)
Graph_GetOnnxIRVersion(apis::OrtApi, graph, onnx_ir_version) = ccall(Base.getproperty(apis, :Graph_GetOnnxIRVersion), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{int64_t}), graph, onnx_ir_version)
Graph_GetNumOperatorSets(apis::OrtApi, graph, num_operator_sets) = ccall(Base.getproperty(apis, :Graph_GetNumOperatorSets), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{size_t}), graph, num_operator_sets)
Graph_GetOperatorSets(apis::OrtApi, graph, domains, opset_versions, num_operator_sets) = ccall(Base.getproperty(apis, :Graph_GetOperatorSets), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{Cchar}}, Ptr{int64_t}, size_t), graph, domains, opset_versions, num_operator_sets)
Graph_GetNumInputs(apis::OrtApi, graph, num_inputs) = ccall(Base.getproperty(apis, :Graph_GetNumInputs), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{size_t}), graph, num_inputs)
Graph_GetInputs(apis::OrtApi, graph, inputs, num_inputs) = ccall(Base.getproperty(apis, :Graph_GetInputs), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtValueInfo}}, size_t), graph, inputs, num_inputs)
Graph_GetNumOutputs(apis::OrtApi, graph, num_outputs) = ccall(Base.getproperty(apis, :Graph_GetNumOutputs), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{size_t}), graph, num_outputs)
Graph_GetOutputs(apis::OrtApi, graph, outputs, num_outputs) = ccall(Base.getproperty(apis, :Graph_GetOutputs), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtValueInfo}}, size_t), graph, outputs, num_outputs)
Graph_GetNumInitializers(apis::OrtApi, graph, num_initializers) = ccall(Base.getproperty(apis, :Graph_GetNumInitializers), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{size_t}), graph, num_initializers)
Graph_GetInitializers(apis::OrtApi, graph, initializers, num_initializers) = ccall(Base.getproperty(apis, :Graph_GetInitializers), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtValueInfo}}, size_t), graph, initializers, num_initializers)
Graph_GetNumNodes(apis::OrtApi, graph, num_nodes) = ccall(Base.getproperty(apis, :Graph_GetNumNodes), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{size_t}), graph, num_nodes)
Graph_GetNodes(apis::OrtApi, graph, nodes, num_nodes) = ccall(Base.getproperty(apis, :Graph_GetNodes), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtNode}}, size_t), graph, nodes, num_nodes)
Graph_GetParentNode(apis::OrtApi, graph, node) = ccall(Base.getproperty(apis, :Graph_GetParentNode), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtNode}}), graph, node)
Graph_GetGraphView(apis::OrtApi, src_graph, nodes, num_nodes, dst_graph) = ccall(Base.getproperty(apis, :Graph_GetGraphView), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtNode}}, size_t, Ptr{Ptr{OrtGraph}}), src_graph, nodes, num_nodes, dst_graph)
Node_GetId(apis::OrtApi, node, node_id) = ccall(Base.getproperty(apis, :Node_GetId), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, node_id)
Node_GetName(apis::OrtApi, node, node_name) = ccall(Base.getproperty(apis, :Node_GetName), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{Cchar}}), node, node_name)
Node_GetOperatorType(apis::OrtApi, node, operator_type) = ccall(Base.getproperty(apis, :Node_GetOperatorType), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{Cchar}}), node, operator_type)
Node_GetDomain(apis::OrtApi, node, domain_name) = ccall(Base.getproperty(apis, :Node_GetDomain), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{Cchar}}), node, domain_name)
Node_GetSinceVersion(apis::OrtApi, node, since_version) = ccall(Base.getproperty(apis, :Node_GetSinceVersion), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Cint}), node, since_version)
Node_GetNumInputs(apis::OrtApi, node, num_inputs) = ccall(Base.getproperty(apis, :Node_GetNumInputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, num_inputs)
Node_GetInputs(apis::OrtApi, node, inputs, num_inputs) = ccall(Base.getproperty(apis, :Node_GetInputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtValueInfo}}, size_t), node, inputs, num_inputs)
Node_GetNumOutputs(apis::OrtApi, node, num_outputs) = ccall(Base.getproperty(apis, :Node_GetNumOutputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, num_outputs)
Node_GetOutputs(apis::OrtApi, node, outputs, num_outputs) = ccall(Base.getproperty(apis, :Node_GetOutputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtValueInfo}}, size_t), node, outputs, num_outputs)
Node_GetNumImplicitInputs(apis::OrtApi, node, num_implicit_inputs) = ccall(Base.getproperty(apis, :Node_GetNumImplicitInputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, num_implicit_inputs)
Node_GetImplicitInputs(apis::OrtApi, node, implicit_inputs, num_implicit_inputs) = ccall(Base.getproperty(apis, :Node_GetImplicitInputs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtValueInfo}}, size_t), node, implicit_inputs, num_implicit_inputs)
Node_GetNumAttributes(apis::OrtApi, node, num_attributes) = ccall(Base.getproperty(apis, :Node_GetNumAttributes), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, num_attributes)
Node_GetAttributes(apis::OrtApi, node, attributes, num_attributes) = ccall(Base.getproperty(apis, :Node_GetAttributes), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtOpAttr}}, size_t), node, attributes, num_attributes)
Node_GetAttributeByName(apis::OrtApi, node, attribute_name, attribute) = ccall(Base.getproperty(apis, :Node_GetAttributeByName), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Cchar}, Ptr{Ptr{OrtOpAttr}}), node, attribute_name, attribute)
OpAttr_GetTensorAttributeAsOrtValue(apis::OrtApi, attribute, attr_tensor) = ccall(Base.getproperty(apis, :OpAttr_GetTensorAttributeAsOrtValue), OrtStatusPtr, (Ptr{OrtOpAttr}, Ptr{Ptr{OrtValue}}), attribute, attr_tensor)
OpAttr_GetType(apis::OrtApi, attribute, type) = ccall(Base.getproperty(apis, :OpAttr_GetType), OrtStatusPtr, (Ptr{OrtOpAttr}, Ptr{OrtOpAttrType}), attribute, type)
OpAttr_GetName(apis::OrtApi, attribute, name) = ccall(Base.getproperty(apis, :OpAttr_GetName), OrtStatusPtr, (Ptr{OrtOpAttr}, Ptr{Ptr{Cchar}}), attribute, name)
Node_GetNumSubgraphs(apis::OrtApi, node, num_subgraphs) = ccall(Base.getproperty(apis, :Node_GetNumSubgraphs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{size_t}), node, num_subgraphs)
Node_GetSubgraphs(apis::OrtApi, node, subgraphs, num_subgraphs, attribute_names) = ccall(Base.getproperty(apis, :Node_GetSubgraphs), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtGraph}}, size_t, Ptr{Ptr{Cchar}}), node, subgraphs, num_subgraphs, attribute_names)
Node_GetGraph(apis::OrtApi, node, graph) = ccall(Base.getproperty(apis, :Node_GetGraph), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{OrtGraph}}), node, graph)
Node_GetEpName(apis::OrtApi, node, out) = ccall(Base.getproperty(apis, :Node_GetEpName), OrtStatusPtr, (Ptr{OrtNode}, Ptr{Ptr{Cchar}}), node, out)
ReleaseExternalInitializerInfo(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseExternalInitializerInfo), Cvoid, (Ptr{OrtExternalInitializerInfo}), input)
ExternalInitializerInfo_GetFilePath(apis::OrtApi, info) = ccall(Base.getproperty(apis, :ExternalInitializerInfo_GetFilePath), Ptr{wchar_t}, (Ptr{OrtExternalInitializerInfo}), info)
ExternalInitializerInfo_GetFileOffset(apis::OrtApi, info) = ccall(Base.getproperty(apis, :ExternalInitializerInfo_GetFileOffset), int64_t, (Ptr{OrtExternalInitializerInfo}), info)
ExternalInitializerInfo_GetByteSize(apis::OrtApi, info) = ccall(Base.getproperty(apis, :ExternalInitializerInfo_GetByteSize), size_t, (Ptr{OrtExternalInitializerInfo}), info)
GetRunConfigEntry(apis::OrtApi, options, config_key) = ccall(Base.getproperty(apis, :GetRunConfigEntry), Ptr{Cchar}, (Ptr{OrtRunOptions}, Ptr{Cchar}), options, config_key)
EpDevice_MemoryInfo(apis::OrtApi, ep_device, memory_type) = ccall(Base.getproperty(apis, :EpDevice_MemoryInfo), Ptr{OrtMemoryInfo}, (Ptr{OrtEpDevice}, OrtDeviceMemoryType), ep_device, memory_type)
CreateSharedAllocator(apis::OrtApi, env, ep_device, mem_type, allocator_type, allocator_options, allocator) = ccall(Base.getproperty(apis, :CreateSharedAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtEpDevice}, OrtDeviceMemoryType, OrtAllocatorType, Ptr{OrtKeyValuePairs}, Ptr{Ptr{OrtAllocator}}), env, ep_device, mem_type, allocator_type, allocator_options, allocator)
GetSharedAllocator(apis::OrtApi, env, mem_info, allocator) = ccall(Base.getproperty(apis, :GetSharedAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtMemoryInfo}, Ptr{Ptr{OrtAllocator}}), env, mem_info, allocator)
ReleaseSharedAllocator(apis::OrtApi, env, ep_device, mem_type) = ccall(Base.getproperty(apis, :ReleaseSharedAllocator), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{OrtEpDevice}, OrtDeviceMemoryType), env, ep_device, mem_type)
GetTensorData(apis::OrtApi, value, out) = ccall(Base.getproperty(apis, :GetTensorData), OrtStatusPtr, (Ptr{OrtValue}, Ptr{Ptr{Cvoid}}), value, out)
GetSessionOptionsConfigEntries(apis::OrtApi, options, out) = ccall(Base.getproperty(apis, :GetSessionOptionsConfigEntries), OrtStatusPtr, (Ptr{OrtSessionOptions}, Ptr{Ptr{OrtKeyValuePairs}}), options, out)
SessionGetMemoryInfoForInputs(apis::OrtApi, session, inputs_memory_info, num_inputs) = ccall(Base.getproperty(apis, :SessionGetMemoryInfoForInputs), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{OrtMemoryInfo}}, size_t), session, inputs_memory_info, num_inputs)
SessionGetMemoryInfoForOutputs(apis::OrtApi, session, outputs_memory_info, num_outputs) = ccall(Base.getproperty(apis, :SessionGetMemoryInfoForOutputs), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{OrtMemoryInfo}}, size_t), session, outputs_memory_info, num_outputs)
SessionGetEpDeviceForInputs(apis::OrtApi, session, inputs_ep_devices, num_inputs) = ccall(Base.getproperty(apis, :SessionGetEpDeviceForInputs), OrtStatusPtr, (Ptr{OrtSession}, Ptr{Ptr{OrtEpDevice}}, size_t), session, inputs_ep_devices, num_inputs)
CreateSyncStreamForEpDevice(apis::OrtApi, ep_device, stream_options, stream) = ccall(Base.getproperty(apis, :CreateSyncStreamForEpDevice), OrtStatusPtr, (Ptr{OrtEpDevice}, Ptr{OrtKeyValuePairs}, Ptr{Ptr{OrtSyncStream}}), ep_device, stream_options, stream)
SyncStream_GetHandle(apis::OrtApi, stream) = ccall(Base.getproperty(apis, :SyncStream_GetHandle), Ptr{Cvoid}, (Ptr{OrtSyncStream}), stream)
ReleaseSyncStream(apis::OrtApi, input) = ccall(Base.getproperty(apis, :ReleaseSyncStream), Cvoid, (Ptr{OrtSyncStream}), input)
CopyTensors(apis::OrtApi, env, src_tensors, dst_tensors, stream, num_tensors) = ccall(Base.getproperty(apis, :CopyTensors), OrtStatusPtr, (Ptr{OrtEnv}, Ptr{Ptr{OrtValue}}, Ptr{Ptr{OrtValue}}, Ptr{OrtSyncStream}, size_t), env, src_tensors, dst_tensors, stream, num_tensors)
Graph_GetModelMetadata(apis::OrtApi, graph, out) = ccall(Base.getproperty(apis, :Graph_GetModelMetadata), OrtStatusPtr, (Ptr{OrtGraph}, Ptr{Ptr{OrtModelMetadata}}), graph, out)
GetModelCompatibilityForEpDevices(apis::OrtApi, ep_devices, num_ep_devices, compatibility_info, out_status) = ccall(Base.getproperty(apis, :GetModelCompatibilityForEpDevices), OrtStatusPtr, (Ptr{Ptr{OrtEpDevice}}, size_t, Ptr{Cchar}, Ptr{OrtCompiledModelCompatibility}), ep_devices, num_ep_devices, compatibility_info, out_status)
CreateExternalInitializerInfo(apis::OrtApi, filepath, file_offset, byte_size, out) = ccall(Base.getproperty(apis, :CreateExternalInitializerInfo), OrtStatusPtr, (Ptr{wchar_t}, int64_t, size_t, Ptr{Ptr{OrtExternalInitializerInfo}}), filepath, file_offset, byte_size, out)

# Export all
for name in names(@__MODULE__; all=true)
    if name in [:eval, :include, Symbol("#eval"), Symbol("#include")]; continue end
    @eval export $name
end

end # module
