use thiserror::Error;

use crate::codegen::tir::Block;

#[derive(Error, Debug)]
pub enum TirError {
    #[error("Block {0} does not end with a terminator")]
    BlockNotTerminated(Block),

    #[error("Function body is empty")]
    EmptyFunctionBody,
}
