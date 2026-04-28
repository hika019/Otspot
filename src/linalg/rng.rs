//! XorShift128+ PRNG + Box–Muller Gaussian sampling (zero-dep)

/// XorShift128+ pseudo-random number generator.
pub struct Xorshift128Plus {
    s: [u64; 2],
}

impl Xorshift128Plus {
    /// Initialise from a single u64 seed. Both state words are guaranteed ≠ 0.
    pub fn new(seed: u64) -> Self {
        let a = (seed ^ 0x9e3779b97f4a7c15u64).max(1);
        let b = (seed.wrapping_mul(0x6c62272e07bb0142u64) ^ 0xbf58476d1ce4e5b9u64).max(1);
        Self { s: [a, b] }
    }

    /// Advance the state and return the next raw u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s[0];
        let y = self.s[1];
        self.s[0] = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        self.s[1] = x;
        x.wrapping_add(y)
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draw two independent N(0,1) samples via Box–Muller.
    pub fn next_gaussian_pair(&mut self) -> (f64, f64) {
        loop {
            let u = self.next_f64();
            if u > 0.0 {
                let v = self.next_f64();
                let r = (-2.0 * u.ln()).sqrt();
                let theta = std::f64::consts::TAU * v;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }
}

/// Fill `buf` with i.i.d. N(0,1) samples.
pub fn fill_gaussian(rng: &mut Xorshift128Plus, buf: &mut [f64]) {
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = rng.next_gaussian_pair();
        buf[i] = a;
        buf[i + 1] = b;
        i += 2;
    }
    if i < buf.len() {
        buf[i] = rng.next_gaussian_pair().0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_nonzero() {
        let mut rng = Xorshift128Plus::new(0);
        for _ in 0..100 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
        }
    }

    #[test]
    fn test_fill_gaussian_finite() {
        let mut rng = Xorshift128Plus::new(12345);
        let mut buf = vec![0.0f64; 101];
        fill_gaussian(&mut rng, &mut buf);
        for &v in &buf {
            assert!(v.is_finite(), "non-finite gaussian sample: {v}");
        }
    }
}
