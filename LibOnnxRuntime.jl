module LibOnnxRuntime

using CEnum: CEnum, @cenum

using Pkg.Artifacts

# TODO - Make arch aware
const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes\\win-x64\\native\\onnxruntime.dll")


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

# Export all
for name in names(@__MODULE__; all=true)
    if name in [:eval, :include, Symbol("#eval"), Symbol("#include")]; continue end
    @eval export $name
end

end # module
