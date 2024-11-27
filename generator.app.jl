using Clang.Generators

cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()
ctx = create_context(["onnxruntimejl.h"], args, options)
build!(ctx)