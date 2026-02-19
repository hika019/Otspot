//! 活性集合管理モジュール
//!
//! Active Set法における活性制約インデックスの管理を行う。
//! Bland則によるサイクル防止をサポートする。

/// 活性制約集合（Working Set）の管理
#[derive(Debug, Clone)]
pub struct WorkingSet {
    /// 活性制約のインデックス（元の制約インデックスで格納、ソート済み）
    indices: Vec<usize>,
}

impl WorkingSet {
    /// 空の活性集合を作成
    #[allow(dead_code)]
    pub fn empty() -> Self {
        WorkingSet { indices: Vec::new() }
    }

    /// 指定インデックスから活性集合を作成
    pub fn from_indices(mut indices: Vec<usize>) -> Self {
        indices.sort_unstable();
        indices.dedup();
        WorkingSet { indices }
    }

    /// 活性制約数を返す
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// 活性集合が空かどうかを返す
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// 活性制約インデックスのスライスを返す
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// 制約が活性かどうかを確認する
    pub fn contains(&self, idx: usize) -> bool {
        self.indices.binary_search(&idx).is_ok()
    }

    /// 制約を活性集合に追加する（Bland則: 最小インデックスを使用）
    pub fn add(&mut self, idx: usize) {
        if let Err(pos) = self.indices.binary_search(&idx) {
            self.indices.insert(pos, idx);
        }
    }

    /// 制約を活性集合から除去する
    pub fn remove(&mut self, idx: usize) {
        if let Ok(pos) = self.indices.binary_search(&idx) {
            self.indices.remove(pos);
        }
    }

    /// k番目の活性制約インデックスを返す（0-indexed）
    pub fn get(&self, k: usize) -> Option<usize> {
        self.indices.get(k).copied()
    }
}
