// Copyright (C) 2023 Afonso Barracha
//
// Rust Keyword Extraction is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Rust Keyword Extraction is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Rust Keyword Extraction. If not, see <http://www.gnu.org/licenses/>.

use crate::tokenizer::to_static_str;
use std::{
    cmp::Ordering,
    collections::{hash_map::RandomState, HashMap, HashSet},
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

#[cfg(not(feature = "parallel"))]
fn basic_sort<'a>(map: &'a HashMap<&str, f32, RandomState>) -> Vec<(&'a &'a str, &'a f32)> {
    let mut map_values = map.iter().collect::<Vec<_>>();
    map_values.sort_unstable_by(|a, b| {
        let order = b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal);

        if order == Ordering::Equal {
            return a.0.cmp(b.0);
        }

        order
    });
    map_values
}

#[cfg(feature = "parallel")]
fn parallel_sort<'a>(map: &'a HashMap<&str, f32, RandomState>) -> Vec<(&'a &'a str, &'a f32)> {
    let mut map_values = map.par_iter().collect::<Vec<_>>();
    map_values.par_sort_unstable_by(|a, b| {
        let order = b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal);

        if order == Ordering::Equal {
            return a.0.cmp(b.0);
        }

        order
    });
    map_values
}

fn sort_ranked_map<'a>(map: &'a HashMap<&'a str, f32, RandomState>) -> Vec<(&'a &'a str, &'a f32)> {
    #[cfg(feature = "parallel")]
    {
        parallel_sort(map)
    }

    #[cfg(not(feature = "parallel"))]
    {
        basic_sort(map)
    }
}

pub fn get_ranked_strings<'a>(
    map: &'a HashMap<&'a str, f32, RandomState>,
    n: usize,
) -> Vec<&'a str> {
    #[cfg(feature = "parallel")]
    {
        sort_ranked_map(map)
            .par_iter()
            .take(n)
            .map(|(&word, _)| word)
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        sort_ranked_map(map)
            .iter()
            .take(n)
            .map(|(&word, _)| word)
            .collect()
    }
}

pub fn get_ranked_scores<'c>(
    map: &'c HashMap<&'c str, f32, RandomState>,
    n: usize,
) -> Vec<(&'c str, f32)> {
    #[cfg(feature = "parallel")]
    {
        sort_ranked_map(map)
            .par_iter()
            .take(n)
            .map(|(&w, v)| (w, **v))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        sort_ranked_map(map)
            .iter()
            .take(n)
            .map(|(&w, v)| (w, **v))
            .collect()
    }
}

pub fn get_special_char_regex() -> Regex {
    Regex::new(r"('s|,|\.)").unwrap()
}

pub fn is_punctuation(word: &str, punctuation: &HashSet<&str>) -> bool {
    word.is_empty() || ((word.graphemes(true).count() == 1) && punctuation.contains(word))
}

pub fn process_word<'a>(
    w: &str,
    special_char_regex: &Regex,
    stopwords: &HashSet<&str>,
    punctuation: &HashSet<&str>,
) -> Option<&'a str> {
    let word = special_char_regex.replace_all(w.trim(), "").to_lowercase();

    if is_punctuation(&word, punctuation) || stopwords.contains(word.as_str()) {
        return None;
    }

    Some(to_static_str(word))
}
