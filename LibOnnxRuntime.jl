module LibOnnxRuntime

using CEnum

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
end

@cenum OrtOpAttrType::UInt32 begin
    ORT_OP_ATTR_UNDEFINED = 0
    ORT_OP_ATTR_INT = 1
    ORT_OP_ATTR_INTS = 2
    ORT_OP_ATTR_FLOAT = 3
    ORT_OP_ATTR_FLOATS = 4
    ORT_OP_ATTR_STRING = 5
    ORT_OP_ATTR_STRINGS = 6
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

mutable struct OrtCUDAProviderOptionsV2 end

mutable struct OrtCANNProviderOptions end

mutable struct OrtDnnlProviderOptions end

mutable struct OrtOp end

mutable struct OrtOpAttr end

mutable struct OrtLogger end

mutable struct OrtShapeInferContext end

mutable struct OrtLoraAdapter end

struct OrtAllocator
    version::UInt32
    Alloc::Ptr{Cvoid}
    Free::Ptr{Cvoid}
    Info::Ptr{Cvoid}
    Reserve::Ptr{Cvoid}
end

# typedef void ( ORT_API_CALL * OrtLoggingFunction
const OrtLoggingFunction = Ptr{Cvoid}

@cenum GraphOptimizationLevel::UInt32 begin
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
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
end

@cenum OrtMemType::Int32 begin
    OrtMemTypeCPUInput = -2
    OrtMemTypeCPUOutput = -1
    OrtMemTypeCPU = -1
    OrtMemTypeDefault = 0
end

@cenum OrtMemoryInfoDeviceType::UInt32 begin
    OrtMemoryInfoDeviceType_CPU = 0
    OrtMemoryInfoDeviceType_GPU = 1
    OrtMemoryInfoDeviceType_FPGA = 2
end

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
    migraphx_int8_enable::Cint
    migraphx_use_native_calibration_table::Cint
    migraphx_int8_calibration_table_name::Ptr{Cchar}
    migraphx_save_compiled_model::Cint
    migraphx_save_model_path::Ptr{Cchar}
    migraphx_load_compiled_model::Cint
    migraphx_load_model_path::Ptr{Cchar}
    migraphx_exhaustive_tune::Bool
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
    data::NTuple{2280, UInt8}
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

mutable struct OrtTrainingApi end

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

@cenum OrtCustomOpInputOutputCharacteristic::UInt32 begin
    INPUT_OUTPUT_REQUIRED = 0
    INPUT_OUTPUT_OPTIONAL = 1
    INPUT_OUTPUT_VARIADIC = 2
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

const ORT_API_VERSION = 20

# Export all
for name in names(@__MODULE__; all=true)
    if name in [:eval, :include, Symbol("#eval"), Symbol("#include")]; continue end
    @eval export $name
end

end # module
