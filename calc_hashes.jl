using Tar, Inflate, SHA

filename = "microsoft.ml.onnxruntime.tar.gz"
println("sha256: ", bytes2hex(open(sha256, filename)))
println("git-tree-sha1: ", Tar.tree_hash(IOBuffer(inflate_gzip(filename))))