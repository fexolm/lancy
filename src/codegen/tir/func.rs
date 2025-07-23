use crate::{support::slotmap::{PrimaryMap}};

use super::{Inst, Block, BlockData};

pub struct Func<I: Inst> {
    blocks: PrimaryMap<Block, BlockData<I>>
}