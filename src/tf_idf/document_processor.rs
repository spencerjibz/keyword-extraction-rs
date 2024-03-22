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

use std::collections::HashSet;

use crate::tokenizer::to_static_str;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::common::{get_special_char_regex, process_word, PUNCTUATION};

pub struct DocumentProcessor<'a> {
    documents: &'a [&'a str],
    stopwords: HashSet<&'a str>,
    punctuation: HashSet<&'a str>,
}

impl<'a> DocumentProcessor<'a> {
    pub fn new(
        documents: &'a [&'a str],
        stopwords: &'a [&'a str],
        punctuation: &'a Option<&'a [&'a str]>,
    ) -> Self {
        Self {
            documents,
            stopwords: stopwords.iter().copied().collect(),
            punctuation: punctuation
                .unwrap_or(&PUNCTUATION)
                .iter()
                .copied()
                .collect(),
        }
    }

    fn process_document<'c>(&self, document: &str, special_char_regex: &Regex) -> &'c str {
        to_static_str(
            document
                .unicode_sentences()
                .map(|s| {
                    s.split_word_bounds()
                        .filter_map(|w| {
                            process_word(w, special_char_regex, &self.stopwords, &self.punctuation)
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<String>>()
                .join(" "),
        )
    }

    pub fn process_documents(&self) -> Vec<&'static str> {
        let special_char_regex = get_special_char_regex();

        #[cfg(feature = "parallel")]
        {
            self.documents
                .par_iter()
                .map(|doc| self.process_document(doc, &special_char_regex))
                .collect::<Vec<_>>()
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.documents
                .iter()
                .map(|doc| self.process_document(doc, &special_char_regex))
                .collect::<Vec<_>>()
        }
    }
}
