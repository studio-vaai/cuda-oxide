// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lints for source items whose only uses disappear with host entry stubs.
use crate::common::has_attr_named;
use proc_macro2::{TokenStream, TokenTree};
use quote::{ToTokens, quote};
use std::collections::BTreeSet;
use syn::{Item, ItemUse, UseTree, parse_quote, spanned::Spanned};

// Splitting a group must not make an unused leaf's diagnostic highlight its
// used siblings. Relocate the path while preserving each token's hygiene.
fn located_at(tokens: TokenStream, location: proc_macro2::Span) -> TokenStream {
    tokens
        .into_iter()
        .map(|mut token| {
            let span = token.span().located_at(location);
            if let TokenTree::Group(group) = &mut token {
                *group = proc_macro2::Group::new(
                    group.delimiter(),
                    located_at(group.stream(), location),
                );
            }
            token.set_span(span);
            token
        })
        .collect()
}

fn identifiers(tokens: TokenStream, names: &mut BTreeSet<String>) {
    for token in tokens {
        match token {
            TokenTree::Ident(name) => {
                names.insert(name.to_string());
            }
            TokenTree::Group(group) => identifiers(group.stream(), names),
            _ => {}
        }
    }
}

fn flatten(tree: &UseTree, prefix: TokenStream, leaves: &mut Vec<(UseTree, Option<String>)>) {
    match tree {
        UseTree::Path(path) => {
            let name = &path.ident;
            flatten(&path.tree, quote!(#prefix #name::), leaves);
        }
        UseTree::Group(group) => {
            for child in &group.items {
                flatten(child, prefix.clone(), leaves);
            }
        }
        UseTree::Name(name) => {
            let imported = if name.ident == "self" {
                prefix
                    .clone()
                    .into_iter()
                    .filter_map(|token| match token {
                        TokenTree::Ident(name) => Some(name.to_string()),
                        _ => None,
                    })
                    .last()
                    .unwrap_or_else(|| "self".into())
            } else {
                name.ident.to_string()
            };
            leaves.push((
                syn::parse2(located_at(quote!(#prefix #tree), tree.span())).unwrap(),
                Some(imported),
            ));
        }
        UseTree::Rename(rename) => {
            leaves.push((
                syn::parse2(located_at(quote!(#prefix #tree), rename.rename.span())).unwrap(),
                Some(rename.rename.to_string()),
            ));
        }
        UseTree::Glob(_) => leaves.push((
            syn::parse2(located_at(quote!(#prefix #tree), tree.span())).unwrap(),
            None,
        )),
    }
}

fn erased_identifiers(items: &[Item], erased: &mut BTreeSet<String>) {
    for item in items {
        match item {
            Item::Fn(function) if has_attr_named(&function.attrs, "kernel") => {
                identifiers(function.block.to_token_stream(), erased)
            }
            // A child may consume an import inherited via `use super::*`.
            Item::Mod(child) => {
                if let Some((_, items)) = &child.content {
                    erased_identifiers(items, erased);
                }
            }
            _ => {}
        }
    }
    // A macro invoked only from erased entries can itself consume imports.
    loop {
        let before = erased.len();
        for item in items {
            if let Item::Macro(mac) = item
                && mac
                    .ident
                    .as_ref()
                    .is_some_and(|name| erased.contains(&name.to_string()))
            {
                identifiers(mac.mac.tokens.clone(), erased);
            }
        }
        if before == erased.len() {
            break;
        }
    }
}

pub(super) fn retain_source_lints(items: &mut Vec<Item>) {
    let mut erased = BTreeSet::new();
    erased_identifiers(items, &mut erased);
    if erased.is_empty() {
        return;
    }
    let mut rewritten = Vec::new();
    for item in items.drain(..) {
        match item {
            Item::Use(import) => {
                let mut leaves = Vec::new();
                flatten(&import.tree, TokenStream::new(), &mut leaves);
                if !leaves
                    .iter()
                    .any(|(_, name)| name.as_ref().is_none_or(|name| erased.contains(name)))
                {
                    rewritten.push(Item::Use(import));
                    continue;
                }
                for (tree, name) in leaves {
                    let mut leaf = ItemUse {
                        tree,
                        ..import.clone()
                    };
                    if name.as_ref().is_some_and(|name| erased.contains(name))
                        || (name.is_none() && !erased.is_empty())
                    {
                        leaf.attrs.push(parse_quote!(#[allow(unused_imports)]));
                    }
                    rewritten.push(Item::Use(leaf));
                }
            }
            Item::Macro(mut mac) => {
                if mac
                    .ident
                    .as_ref()
                    .is_some_and(|name| erased.contains(&name.to_string()))
                {
                    mac.attrs.push(parse_quote!(#[allow(unused_macros)]));
                }
                rewritten.push(Item::Macro(mac));
            }
            Item::Mod(mut child) => {
                if let Some((_, items)) = &mut child.content {
                    retain_source_lints(items);
                }
                rewritten.push(Item::Mod(child));
            }
            other => rewritten.push(other),
        }
    }
    *items = rewritten;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_erased_body_imports_and_macros_receive_lint_allowances() {
        let mut module: syn::ItemMod = parse_quote! {
            mod kernels {
                use helpers::{used, unused, original as renamed};
                macro_rules! launch { () => {} }
                macro_rules! unused_macro { () => {} }
                #[kernel] fn entry() { used(); renamed(); launch!(); }
            }
        };
        let (_, items) = module.content.as_mut().unwrap();
        retain_source_lints(items);
        for item in items {
            match item {
                Item::Use(import) => {
                    let allowed = import.attrs.iter().any(|a| a.path().is_ident("allow"));
                    assert_eq!(
                        allowed,
                        !import
                            .tree
                            .to_token_stream()
                            .to_string()
                            .ends_with("unused")
                    );
                }
                Item::Macro(mac) => {
                    let allowed = mac.attrs.iter().any(|a| a.path().is_ident("allow"));
                    assert_eq!(allowed, mac.ident.as_ref().unwrap() == "launch");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn macro_body_and_self_imports_are_tracked_without_suppressing_helper_only_imports() {
        let mut module: syn::ItemMod = parse_quote! {
            mod kernels {
                use helpers::{self, unused};
                use unrelated::{one, two};
                macro_rules! invoke { () => { helpers::run(); } }
                #[kernel] fn entry() { invoke!(); }
                mod child { fn helper() { let unused = 0; } }
            }
        };
        let (_, items) = module.content.as_mut().unwrap();
        retain_source_lints(items);
        let imports: Vec<_> = items
            .iter()
            .filter_map(|item| {
                if let Item::Use(import) = item {
                    Some(import)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(imports.len(), 3);
        assert!(imports[0].attrs.iter().any(|a| a.path().is_ident("allow")));
        assert!(imports[1].attrs.is_empty());
        assert!(imports[2].attrs.is_empty());
        assert!(matches!(imports[2].tree, UseTree::Path(_)));
    }
}
