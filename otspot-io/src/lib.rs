#![deny(clippy::print_stdout, clippy::print_stderr)]
//! File I/O for the otspot solver — MPS, QPS, and QPLIB format parsers.

pub mod mps;
pub mod qps;
pub mod qplib;

mod common;

/// Thread-local peak-allocation tracker for memory sentinel tests.
///
/// Mirrors the implementation in `otspot-core` so that `otspot-io`'s own
/// qplib memory tests can use the same tracking allocator without a
/// cross-crate dependency on a `cfg(test)`-only module.
#[cfg(test)]
pub(crate) mod peak_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static CURRENT: Cell<isize> = const { Cell::new(0) };
        static BASELINE: Cell<isize> = const { Cell::new(0) };
        static PEAK_DELTA: Cell<isize> = const { Cell::new(0) };
    }

    pub fn begin() {
        CURRENT.with(|c| BASELINE.with(|b| b.set(c.get())));
        PEAK_DELTA.with(|p| p.set(0));
    }

    pub fn peak_bytes() -> usize {
        PEAK_DELTA.with(|p| p.get().max(0) as usize)
    }

    pub fn current_bytes() -> isize {
        CURRENT.with(|c| c.get()) - BASELINE.with(|b| b.get())
    }

    #[inline]
    fn update(delta: isize) {
        CURRENT.with(|c| {
            let new = c.get() + delta;
            c.set(new);
            let above = new - BASELINE.with(|b| b.get());
            PEAK_DELTA.with(|p| {
                if above > p.get() {
                    p.set(above);
                }
            });
        });
    }

    pub struct TrackingAlloc;

    unsafe impl GlobalAlloc for TrackingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                update(layout.size() as isize);
            }
            ptr
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc_zeroed(layout);
            if !ptr.is_null() {
                update(layout.size() as isize);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
            update(-(layout.size() as isize));
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new_ptr = System.realloc(ptr, layout, new_size);
            if !new_ptr.is_null() {
                update(new_size as isize - layout.size() as isize);
            }
            new_ptr
        }
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOC: peak_alloc::TrackingAlloc = peak_alloc::TrackingAlloc;
