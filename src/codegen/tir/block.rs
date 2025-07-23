use crate::slotmap_key;

use super::Inst;
slotmap_key!(Block(u16));

pub struct BlockData<I: Inst> {
    insts: Vec<I>,
}