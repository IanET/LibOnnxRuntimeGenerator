using Pkg.Artifacts

@static if Sys.iswindows()
    const ORT_CHAR_T = Cwchar_t
elseif Sys.islinux()
    const ORT_CHAR_T = Cchar
end

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

const size_t = Csize_t
const int64_t = Clonglong
const uint64_t = Culonglong
const int32_t = Cint
const uint32_t = Cuint
const int16_t = Cshort
const uint16_t = Cushort
const int8_t = Cchar
const uint8_t = Cuchar
const byte = UInt8
