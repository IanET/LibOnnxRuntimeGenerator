# 
# Automatically generated file - do not edit
#

onnxruntime = joinpath(@__DIR__, "onnxruntime.dll")

function __init__()
    # TODO - make a proper 'artifact'
    chmod(onnxruntime, filemode(onnxruntime) | 0o755) # dll needs to executable
end
