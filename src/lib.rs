#![deny(clippy::all)]

#![warn(clippy::pedantic)]

#![allow(
    clippy::cast_possible_truncation,
    clippy::iter_without_into_iter,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::manual_let_else,
    clippy::items_after_statements,
    clippy::cast_possible_wrap,
    clippy::match_same_arms,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::implicit_hasher,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::unreadable_literal,
)]
pub mod codegen;
pub mod support;
