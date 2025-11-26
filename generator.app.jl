using Clang.Generators
using Clang.LibClang

cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()

# AI Generated code for the OrtApi struct to create Julia wrappers for each function pointer.
# function gen_api_function(io::IO, name, return_type, arg_types, arg_names)
#     # Map C types to Julia types
#     julia_return_type = Generators.translate(return_type)
#     julia_arg_types = [Generators.translate(t) for t in arg_types]

#     # Create argument list with type annotations
#     julia_args = ["$(arg_name)::$(arg_type)" for (arg_name, arg_type) in zip(arg_names, julia_arg_types)]
#     prepend!(julia_args, ["ort::Ptr{OrtApi}"])

#     println(io, "function $(name)(", join(julia_args, ", "), ")")
#     print(io, "    ccall(Base.getproperty(ort, :$(name)), $(julia_return_type), (")
#     print(io, join(julia_arg_types, ", "), "), ")
#     println(io, join(arg_names, ", "), ")") # Don't pass `ort` to ccall
#     println(io, "end")
#     println(io)
# end

# function rewriter(io::IO, node::Generators.StructDefinition, options)
#     if node.name != "OrtApi"
#         # If it's not the struct we want to customize, print the default
#         Generators.pretty_print(io, node, options)
#         return
#     end

#     # It is the OrtApi struct, so run our custom logic
#     # First, generate the default struct definition
#     Generators.pretty_print(io, node, options)

#     # Then, generate wrappers for each function pointer
#     for field in node.fields
#         name = field.name
#         func_ptr_type = field.type
#         # ... (rest of the logic is the same as in common_utils.jl)
#     end
# end

# function rewrite(expr::Expr)
#     @info "Expr type: $expr"
#     # if expr isa Generators.StructDefinition
#     #     @info "Rewriting struct: $(expr.name)"
#     # end
# end

function gen_api_function(io::IO, struct_name::String, function_name::String, return_type::AbstractJuliaType, arg_types::Vector{AbstractJuliaType}, arg_names::Vector{String})

    # Convert to Julia type strings
    jrt = Generators.translate(return_type) 
    jats = [Generators.translate(t) for t in arg_types]
    # julia_args = ["$(arg_name)::$(arg_type)" for (arg_name, arg_type) in zip(arg_names, arg_types)]
    # prepend!(julia_args, ["pstrct::Ptr{$(struct_name)}"])
    julia_args = ["pstrct::Ptr{$(struct_name)}", arg_names...]

    println(io, "function $(function_name)(", join(julia_args, ", "), ")")
    print(io, "    ccall(Base.getproperty(pstrct, :$(function_name)), $(jrt), (")
    print(io, join(jats, ", "), "), ")
    println(io, join(arg_names, ", "), ")")
    println(io, "end")
    println(io)
end

function rewrite(dag::ExprDAG)
    struct_sym = :OrtApi # Test
    n = findfirst(n->n.id == struct_sym, ctx.dag.nodes)
    node = ctx.dag.nodes[n]
    c = node.cursor
    t = Clang.getCursorType(c)
    fldc = fields(t)
    
    fc = fldc[1]

    ft = Clang.getCursorType(fc)
    function_name = spelling(fc)
    pt = Clang.getPointeeType(ft)
    @assert kind(pt) == CXType_FunctionProto

    rt = clang_getResultType(pt)
    return_type = CLType(rt) |> tojulia
    # crt = clang_getCanonicalType(return_type)

    na = clang_getNumArgTypes(pt)
    ats = [clang_getArgType(pt, i) for i in 0:na-1]
    arg_type = CLType.(ats) .|> tojulia
    ancs = filter(c -> kind(c) == CXCursor_ParmDecl, children(fc))
    arg_names = spelling.(ancs)
    
    # TODO - generate the function wrapper here
    @info "Function: $function_name"
    @info "  Return type: $return_type" 
    @info "  Arg types: $arg_type"
    @info "  Arg names: $arg_names"
    gen_api_function(stdout, String(struct_sym), function_name, return_type, arg_type, arg_names)
end

ctx = create_context(["onnxruntimejl.h"], args, options)
build!(ctx, BUILDSTAGE_NO_PRINTING)
rewrite(ctx.dag)
build!(ctx, BUILDSTAGE_PRINTING_ONLY)