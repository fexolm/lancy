mod reg;
mod block;
mod inst;
mod func;

pub use reg::*;
pub use block::*;
pub use inst::*;
pub use func::*;

use crate::{support::slotmap::{PrimaryMap, SecondaryMap}};

