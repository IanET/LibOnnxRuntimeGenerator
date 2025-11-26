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

function rewrite(dag::ExprDAG)
    n = findfirst(n->n.id == :OrtApi, ctx.dag.nodes)
    node = ctx.dag.nodes[n]
    c = node.cursor
    t = Clang.getCursorType(c)
    fldc = fields(t)
    
    field_cursor = fldc[1]
    ft = Clang.getCursorType(field_cursor)
    fn = spelling(field)
    pt = Clang.getPointeeType(ft)
    rt = clang_getResultType(pt)
    jrt = CLType(rt) |> tojulia
    crt = clang_getCanonicalType(rt)

    na = clang_getNumArgTypes(pt)
    ats = [clang_getArgType(pt, i) for i in 0:na-1]
    jats = CLType.(ats) .|> tojulia
    arg_name_cursors = filter(c -> kind(c) == CXCursor_ParmDecl, children(field_cursor))
    arg_names = spelling.(arg_name_cursors)
    
    # todo
    
end

ctx = create_context(["onnxruntimejl.h"], args, options)
build!(ctx, BUILDSTAGE_NO_PRINTING)
rewrite(ctx.dag)
build!(ctx, BUILDSTAGE_PRINTING_ONLY)