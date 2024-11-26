using Pkg.Artifacts

# TODO - Make arch aware
const OnnxRuntime = joinpath(artifact"OnnxRuntime", "runtimes\\win-x64\\native\\onnxruntime.dll")
