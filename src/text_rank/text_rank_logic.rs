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

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct TextRankLogic;

fn score_phrase<'c>(phrase: &'c str, word_rank: &HashMap<&'c str, f32>) -> (&'c str, f32) {
    let words = phrase.split_whitespace().collect::<Vec<&str>>();
    let score = words
        .iter()
        .filter_map(|word| word_rank.get(*word))
        .sum::<f32>();

    (phrase, score / words.len() as f32)
}

fn score_word(
    edges: &HashMap<&str, f32>,
    node_indexes: &HashMap<&str, usize>,
    outgoing_weight_sums: &HashMap<&str, f32>,
    prev_scores: &[f32],
    damping: f32,
) -> f32 {
    let new_score = edges
        .iter()
        .map(|(neighbor, weight)| {
            let neighbor_index = node_indexes[neighbor];
            let neighbor_outgoing_sum = outgoing_weight_sums[neighbor];
            weight / neighbor_outgoing_sum * prev_scores[neighbor_index]
        })
        .sum::<f32>();

    (1.0 - damping) + damping * new_score
}

fn get_node_indexes<'a>(nodes: &[&&'a str]) -> HashMap<&'a str, usize> {
    #[cfg(feature = "parallel")]
    {
        nodes
            .par_iter()
            .enumerate()
            .map(|(i, &&w)| (w, i))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        nodes
            .iter()
            .enumerate()
            .map(|(i, &&w)| (w, i))
            .collect()
    }
}

fn get_scores(
    graph: &HashMap<&str, HashMap<&str, f32>>,
    node_indexes: &HashMap<&str, usize>,
    outgoing_weight_sums: &HashMap<&str, f32>,
    prev_scores: &[f32],
    damping: f32,
) -> Vec<f32> {
    #[cfg(feature = "parallel")]
    {
        graph
            .par_iter()
            .map(|(_, edges)| {
                score_word(
                    edges,
                    node_indexes,
                    outgoing_weight_sums,
                    prev_scores,
                    damping,
                )
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        graph
            .values()
            .map(|edges| {
                score_word(
                    edges,
                    node_indexes,
                    outgoing_weight_sums,
                    prev_scores,
                    damping,
                )
            })
            .collect()
    }
}

fn check_tolorance(scores: &[f32], prev_scores: &[f32], tol: f32) -> bool {
    #[cfg(feature = "parallel")]
    {
        scores.par_iter().enumerate().all(|(i, score)| {
            let prev_score = prev_scores[i];
            (score - prev_score).abs() < tol
        })
    }

    #[cfg(not(feature = "parallel"))]
    {
        scores
            .iter()
            .zip(prev_scores.iter())
            .all(|(score, prev_score)| (score - prev_score).abs() < tol)
    }
}

impl TextRankLogic {
    pub fn build_text_rank<'a>(
        words: &[&'a str],
        phrases: &[&'a str],
        window_size: usize,
        damping: f32,
        tol: f32,
    ) -> (HashMap<&'a str, f32>, HashMap<&'a str, f32>) {
        let word_rank =
            Self::create_word_rank(Self::create_graph(words, window_size), damping, tol);
        let phrase_rank = Self::rank_phrases(phrases, &word_rank);
        (word_rank, phrase_rank)
    }

    fn add_edge<'c>(graph: &mut HashMap<&'c str, HashMap<&'c str, f32>>, word1: &'c str, word2: &'c str) {
        graph
            .entry(word1)
            .or_default()
            .entry(word2)
            .and_modify(|e| *e += 1.0)
            .or_insert(1.0);
    }

    fn create_graph<'a>(
        words: &[&'a str],
        window_size: usize,
    ) -> HashMap<&'a str, HashMap<&'a str, f32>> {
        let mut graph = HashMap::new();

        words
            .iter()
            .enumerate()
            .flat_map(|(i, word1)| {
                words[i + 1..]
                    .iter()
                    .take(window_size)
                    .filter( move |&word2| word1 != word2)
                    .map(move |word2| (word1, word2))
            })
            .for_each(|(word1, word2)| {
                Self::add_edge(&mut graph, word1, word2);
                Self::add_edge(&mut graph, word2, word1);
            });

        graph
    }

    fn get_outgoing_weight_sum<'a>(
        graph: &HashMap<&'a str, HashMap<&str, f32>>,
    ) -> HashMap<&'a str, f32> {
        #[cfg(feature = "parallel")]
        {
            graph
                .par_iter()
                .map(|(&node, edges)| {
                    let outgoing_weight_sum = edges.values().sum();
                    (node, outgoing_weight_sum)
                })
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            graph
                .iter()
                .map(|(&node, edges)| {
                    let outgoing_weight_sum = edges.values().sum();
                    (node, outgoing_weight_sum)
                })
                .collect()
        }
    }

    fn create_word_rank<'c>(
        graph: HashMap<&'c str, HashMap<&str, f32>>,
        damping: f32,
        tol: f32,
    ) -> HashMap<&'c str, f32> {
        let nodes = graph.keys().collect::<Vec<_>>();
        let n = nodes.len();
        let node_indexes = get_node_indexes(&nodes);
        let mut scores = vec![1.0_f32; n];
        let outgoing_weight_sums = Self::get_outgoing_weight_sum(&graph);

        loop {
            let prev_scores = scores.to_owned();
            scores = get_scores(
                &graph,
                &node_indexes,
                &outgoing_weight_sums,
                &prev_scores,
                damping,
            );

            if check_tolorance(&scores, &prev_scores, tol) {
                break;
            }
        }

        #[cfg(feature = "parallel")]
        {
            nodes
                .par_iter()
                .map(|&&node| (node, scores[node_indexes[node]]))
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            nodes
                .iter()
                .map(|&&node| (node, scores[node_indexes[node]]))
                .collect()
        }
    }

    fn rank_phrases<'c>(
        phrases: &[&'c str],
        word_scores: &HashMap<&'c str, f32>,
    ) -> HashMap<&'c str, f32> {
        #[cfg(feature = "parallel")]
        {
            phrases
                .par_iter()
                .map(|phrase| score_phrase(phrase, word_scores))
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            phrases
                .iter()
                .map(|phrase| score_phrase(phrase, word_scores))
                .collect()
        }
    }
}
