use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error("failed to parse LLVM IR: {0}")]
    Parse(String),
    #[error("function `{0}` not found in module")]
    FunctionNotFound(String),
    #[error("unsupported LLVM construct: {0}")]
    Unsupported(String),
    #[error("malformed LLVM IR: {0}")]
    Malformed(String),
    #[error("JIT error: {0}")]
    Jit(#[from] std::io::Error),
}
