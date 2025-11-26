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