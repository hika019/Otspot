//! Mehrotra IPM (IP-PMM)。
//! 1 層 retry で eps を直線厳格化、status 変換は API 境界の 1 箇所に集約、KKT は元空間判定。

pub mod outcome;
pub mod kkt;
pub mod core;
pub mod attempt;

pub use attempt::solve_ipm;
