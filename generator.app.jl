using Clang
using Clang.Generators
using Clang.LibClang

cd(@__DIR__)

options = load_options(joinpath(@__DIR__, "generator.toml"))
args = get_default_args()

mutable struct StructApiPrinter <: Clang.Generators.AbstractPrinter end

function gen_api_function(io::IO, options::Dict, struct_name::String, function_name::String, return_type::AbstractJuliaType, arg_types::Vector, arg_names::Vector)
    jrt = Generators.translate(return_type, options["codegen"]) 
    jats = [Generators.translate(t, options["codegen"]) for t in arg_types]
    julia_args = ["apis::$(struct_name)", arg_names...]

    print(io, "$(function_name)(", join(julia_args, ", "), ") = ")
    print(io, "ccall(Base.getproperty(apis, :$(function_name)), $(jrt), (")
    if length(jats) == 1
        print(io, jats[1], ",), ")
    else
        print(io, join(jats, ", "), "), ")
    end
    println(io, join(arg_names, ", "), ")")
end

function rewrite(io::IO, options::Dict, struct_sym::Symbol, fc::CLFieldDecl)
    ft = Clang.getCursorType(fc)
    function_name = spelling(fc)
    pt = Clang.getPointeeType(ft)

    if kind(pt) != CXType_FunctionProto && kind(pt) != CXType_FunctionNoProto
        @info "Skipping field '$function_name', not function pointer type"
        return
    end

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

    gen_api_function(io, options, String(struct_sym), function_name, return_type, arg_types, arg_names)
end

function rewrite(io, dag, options, struct_name)
    struct_sym = Symbol(struct_name)
    n = findfirst(n->n.id == struct_sym, dag.nodes)
    if n === nothing
        @warn "Struct $struct_sym not found in DAG"
        return
    end
    node = dag.nodes[n]
    c = node.cursor
    t = Clang.getCursorType(c)
    fldc = fields(t)
    for fc in fldc
        rewrite(io, options, struct_sym, fc)
    end
end

function (x::StructApiPrinter)(dag::ExprDAG, options::Dict)
    file = options["general"]["output_file_path"]
    codegen_options = get(options, "codegen", Dict())
    wrap_api_structs = get(codegen_options, "wrap_api_structs", String[])
    codegen_options["DAG_tags"] = dag.tags
    codegen_options["DAG_ids"] = dag.ids
    codegen_options["DAG_ids_extra"] = dag.ids_extra

    if !isempty(wrap_api_structs)
        open(file, "a") do io
            for struct_name in wrap_api_structs
                @info "Wrapping API struct: $struct_name"
                rewrite(io, dag, options, struct_name)
                println(io)
            end
        end
    end

    delete!(codegen_options, "DAG_tags")
    delete!(codegen_options, "DAG_ids")
    delete!(codegen_options, "DAG_ids_extra")

    return dag
end

ctx = create_context(["onnxruntimejl.h"], args, options)

epilog_idx = findfirst(p -> p isa Clang.Generators.EpiloguePrinter, ctx.passes)
insert!(ctx.passes, epilog_idx, StructApiPrinter())

build!(ctx)
