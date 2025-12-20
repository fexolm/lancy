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
)]
pub mod codegen;
pub mod support;
