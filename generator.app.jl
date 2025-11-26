using Clang
using Clang.Generators
using Clang.LibClang

cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()

function gen_api_function(io::IO, struct_name::String, function_name::String, return_type::AbstractJuliaType, arg_types::Vector, arg_names::Vector)
    jrt = Generators.translate(return_type) 
    jats = [Generators.translate(t) for t in arg_types]
    julia_args = ["pstrct::Ptr{$(struct_name)}", arg_names...]

    print(io, "$(function_name)(", join(julia_args, ", "), ") = ")
    print(io, "ccall(Base.getproperty(pstrct, :$(function_name)), $(jrt), (")
    print(io, join(jats, ", "), "), ")
    println(io, join(arg_names, ", "), ")")
    println(io)
end

function rewrite(io::IO, struct_sym::Symbol, fc::CLFieldDecl)
    ft = Clang.getCursorType(fc)
    function_name = spelling(fc)
    pt = Clang.getPointeeType(ft)
    @assert kind(pt) == CXType_FunctionProto || kind(pt) == CXType_FunctionNoProto "Expected function pointer type, got $(kind(pt)) for $function_name"

    rt = clang_getResultType(pt)
    return_type = CLType(rt) |> tojulia

    na = clang_getNumArgTypes(pt)
    ats = [clang_getArgType(pt, i) for i in 0:na-1]
    arg_types = CLType.(ats) .|> tojulia
    ancs = filter(c -> kind(c) == CXCursor_ParmDecl, children(fc))
    arg_names = spelling.(ancs)
    
    # @info "Function: $function_name"
    # @info "  Return type: $return_type" 
    # @info "  Arg names: $arg_names"
    # @info "  Arg types: $arg_types"

    gen_api_function(io, String(struct_sym), function_name, return_type, arg_types, arg_names)
end

function rewrite(dag::ExprDAG)
    struct_sym = :OrtApi # Test
    n = findfirst(n->n.id == struct_sym, dag.nodes)
    node = dag.nodes[n]
    c = node.cursor
    t = Clang.getCursorType(c)
    fldc = fields(t)
    for fc in fldc
        rewrite(stdout, struct_sym, fc)
    end
end

ctx = create_context(["onnxruntimejl.h"], args, options)
build!(ctx, BUILDSTAGE_NO_PRINTING)
rewrite(ctx.dag)
build!(ctx, BUILDSTAGE_PRINTING_ONLY)