/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Number the SDK's generated local values and blocks in LLVM slot order.
//!
//! The exporter reserves names before printing PHIs, hoisted allocas and
//! pointer adapters. Their original IDs therefore are not LLVM slot IDs.
//! Number the completed function in definition order, then rewrite its uses.
//! This consumes our own emitted syntax, not arbitrary external LLVM IR.

use rustc_hash::FxHashMap;
use std::fmt::Write;

fn identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"_.$-".contains(&byte)
}

/// Local identifier tokens, excluding strings and comments (notably asm).
struct Locals<'a> {
    text: &'a str,
    cursor: usize,
}

impl<'a> Iterator for Locals<'a> {
    type Item = (usize, usize, &'a str);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        while self.cursor < bytes.len() {
            let start = self.cursor;
            self.cursor += 1;
            match bytes[start] {
                b'"' => {
                    while self.cursor < bytes.len() {
                        let byte = bytes[self.cursor];
                        self.cursor += 1;
                        if byte == b'\\' && self.cursor < bytes.len() {
                            self.cursor += 1;
                        } else if byte == b'"' {
                            break;
                        }
                    }
                }
                b';' => {
                    while self.cursor < bytes.len() && bytes[self.cursor] != b'\n' {
                        self.cursor += 1;
                    }
                }
                b'%' => {
                    let name_start = self.cursor;
                    while self.cursor < bytes.len() && identifier_byte(bytes[self.cursor]) {
                        self.cursor += 1;
                    }
                    if self.cursor > name_start {
                        return Some((start, self.cursor, &self.text[name_start..self.cursor]));
                    }
                }
                _ => {}
            }
        }
        None
    }
}

fn label(line: &str) -> Option<&str> {
    let name = line.trim().strip_suffix(':')?;
    (!name.is_empty() && name.bytes().all(identifier_byte)).then_some(name)
}

pub(super) fn number_function_locals(
    function: &str,
    argument_count: usize,
) -> Result<String, String> {
    let mut slots = FxHashMap::default();
    let mut add = |name| {
        let next = slots.len();
        if slots.insert(name, next).is_some() {
            return Err(format!("duplicate generated local identifier %{name}"));
        }
        Ok(())
    };
    let mut lines = function.lines();
    let header = lines.next().ok_or("empty generated function")?;
    let mut arguments_seen = 0;
    for (_, _, name) in (Locals {
        text: header,
        cursor: 0,
    }) {
        if name
            .strip_prefix('v')
            .and_then(|s| s.parse::<usize>().ok())
            .is_some_and(|index| index < argument_count)
        {
            add(name)?;
            arguments_seen += 1;
        }
    }
    if arguments_seen != argument_count {
        return Err(format!(
            "expected {argument_count} generated function arguments, found {arguments_seen}"
        ));
    }
    for line in lines {
        if let Some(name) = label(line) {
            add(name)?;
        } else if let Some(rest) = line.trim_start().strip_prefix('%')
            && let Some((name, _)) = rest.split_once(" = ")
        {
            if !name.bytes().all(identifier_byte) {
                return Err(format!("invalid generated local identifier %{name}"));
            }
            add(name)?;
        }
    }
    // Types are printed inline by this exporter. Global names and metadata
    // never enter this function-local map.
    let mut result = String::with_capacity(function.len());
    for line in function.split_inclusive('\n') {
        if let Some(name) = label(line) {
            write!(result, "{}:", slots[name]).unwrap();
            if line.ends_with('\n') {
                result.push('\n');
            }
            continue;
        }
        let mut copied = 0;
        for (start, end, name) in (Locals {
            text: line,
            cursor: 0,
        }) {
            if let Some(slot) = slots.get(name) {
                result.push_str(&line[copied..start]);
                write!(result, "%{slot}").unwrap();
                copied = end;
            }
        }
        result.push_str(&line[copied..]);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::number_function_locals;

    #[test]
    fn slots_follow_printed_definitions_including_loop_backedges() {
        let input = "define i32 @kernel(i32 %v0) {\nentry:\n  %v9 = alloca i32\n  br label %bb0\nbb0:\n  %v1 = phi i32 [ %v0, %entry ], [ %v2, %bb1 ]\n  br label %bb1\nbb1:\n  %v2 = add i32 %v1, 1\n  br label %bb0\n}\n";
        let expected = "define i32 @kernel(i32 %0) {\n1:\n  %2 = alloca i32\n  br label %3\n3:\n  %4 = phi i32 [ %0, %1 ], [ %6, %5 ]\n  br label %5\n5:\n  %6 = add i32 %4, 1\n  br label %3\n}\n";
        assert_eq!(number_function_locals(input, 1).unwrap(), expected);
    }

    #[test]
    fn preserve_asm_comments_symbols_and_metadata() {
        let input = "define i32 @v0(i32 %v0) {\nentry:\n  ; %v0 and entry: stay readable\n  %v1 = call i32 asm \"mov.u32 $0, %v0; \\22quoted\\22\", \"=r\"(), !annotation !0\n  call void @entry(i32 %v1)\n  ret i32 %v0\n}\n";
        let actual = number_function_locals(input, 1).unwrap();
        assert!(actual.contains("@v0(i32 %0)"));
        assert!(actual.contains("; %v0 and entry: stay readable"));
        assert!(actual.contains("\"mov.u32 $0, %v0; \\22quoted\\22\""));
        assert!(actual.contains("@entry(i32 %2)"));
        assert!(actual.contains("!annotation !0"));
    }
}
