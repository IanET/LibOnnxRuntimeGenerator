using Tar, Inflate, SHA

# NB Don't use Windows GUI to create tar.gz, it'll be missing execute on the dll. 
# Use WSL eg
#   tar -czvf ../microsoft.ml.onnxruntime.tar.gz .

filename = "microsoft.ml.onnxruntime.tar.gz"
println("sha256: ", bytes2hex(open(sha256, filename)))
println("git-tree-sha1: ", Tar.tree_hash(IOBuffer(inflate_gzip(filename))))