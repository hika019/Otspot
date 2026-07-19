use otspot::problem::SolveStatus;
use std::path::Path;
use std::time::Instant;

fn main() {
    let problems = vec![
        ("afiro", "tests/netlib/afiro.mps"),
        ("blend", "tests/netlib/blend.mps"),
        ("kb2", "tests/netlib/kb2.mps"),
        ("adlittle", "tests/netlib/adlittle.mps"),
        ("share2b", "tests/netlib/share2b.mps"),
        ("share1b", "tests/netlib/share1b.mps"),
        ("stocfor1", "tests/netlib/stocfor1.mps"),
        ("brandy", "tests/netlib/brandy.mps"),
        ("scorpion", "tests/netlib/scorpion.mps"),
        ("boeing2", "tests/netlib/boeing2.mps"),
        ("fit1d", "tests/netlib/fit1d.mps"),
    ];

    println!(
        "{:<12} {:>6} {:>6} {:>12} Status",
        "Problem", "Rows", "Cols", "Time"
    );
    println!("{}", "-".repeat(65));

    for (name, path) in &problems {
        if !Path::new(path).exists() {
            println!("{:<12} -- file not found --", name);
            continue;
        }
        let mps_data = std::fs::read_to_string(path).unwrap();
        let lp = otspot::io::mps::parse_mps(&mps_data).unwrap();
        let rows = lp.a.nrows();
        let cols = lp.a.ncols();

        let start = Instant::now();
        let result = otspot::solve(&lp);
        let elapsed = start.elapsed();

        let status = match result.status {
            SolveStatus::Optimal => format!("Optimal({:.6})", result.objective),
            SolveStatus::Infeasible => "Infeasible".to_string(),
            SolveStatus::Unbounded => "Unbounded".to_string(),
            SolveStatus::MaxIterations => "MaxIterations".to_string(),
            SolveStatus::Stalled => "Stalled".to_string(),
            SolveStatus::SuboptimalSolution => "SuboptimalSolution".to_string(),
            SolveStatus::Timeout => "Timeout".to_string(),
            SolveStatus::NumericalError => "NumericalError".to_string(),
            _ => "Unknown".to_string(),
        };

        let time_str = if elapsed.as_secs() > 0 {
            format!("{:.2}s", elapsed.as_secs_f64())
        } else if elapsed.as_millis() > 0 {
            format!("{:.1}ms", elapsed.as_secs_f64() * 1000.0)
        } else {
            format!("{:.1}µs", elapsed.as_secs_f64() * 1_000_000.0)
        };

        println!(
            "{:<12} {:>6} {:>6} {:>12} {}",
            name, rows, cols, time_str, status
        );
    }
}
