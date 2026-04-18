//! Tiny JIT runtime: mmap a RW region, memcpy the code, flip to RX, hand out
//! function pointers.
//!
//! Linux/macOS (POSIX) only. Windows would need `VirtualAlloc` /
//! `VirtualProtect` — punted until we need it.

use std::ffi::c_void;
use std::io;
use std::ptr;

/// A JIT-loaded code region. `Drop` munmaps.
pub struct Module {
    code: *mut u8,
    size: usize,
}

// The mapped region is not `Send` by default because of the raw pointer. The
// memory itself is safe to move between threads (we never mutate it after
// mprotect). Keep it simple: don't implement Send/Sync.

impl Module {
    /// Map `bytes.len()` bytes of anonymous memory as RW, copy `bytes` in,
    /// flip the protection to RX, and return the loaded module.
    ///
    /// # Errors
    /// Propagates the last OS error from `mmap` / `mprotect`.
    pub fn load(bytes: &[u8]) -> io::Result<Self> {
        assert!(!bytes.is_empty(), "cannot load an empty code region");
        // Round up to a page so mprotect covers the whole thing.
        let page = page_size();
        let size = bytes.len().div_ceil(page) * page;

        // SAFETY: We pass nul base, valid size, correct prot/flags. The kernel
        // either returns a valid mapping or MAP_FAILED; we check.
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `ptr` is a fresh anonymous mapping of `size` bytes, `bytes`
        // fits by construction (size ≥ bytes.len()), and the two regions do
        // not overlap.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
        }

        // Flip to executable. After this, further writes would trap (W^X).
        // SAFETY: `ptr` / `size` came from `mmap` just above.
        let rc = unsafe { libc::mprotect(ptr, size, libc::PROT_READ | libc::PROT_EXEC) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            // SAFETY: we own the mapping.
            unsafe {
                libc::munmap(ptr, size);
            }
            return Err(err);
        }

        Ok(Self {
            code: ptr.cast::<u8>(),
            size,
        })
    }

    /// Returns a raw, non-null pointer to the start of the loaded code.
    #[must_use]
    pub fn code_ptr(&self) -> *const u8 {
        self.code
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Cast the entry point to a typed function pointer.
    ///
    /// # Safety
    /// Caller guarantees that the emitted machine code's signature matches
    /// `F` (both ABI and argument/return types). Mismatches are UB.
    #[must_use]
    pub unsafe fn entry<F: Copy>(&self) -> F {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const ()>(),
            "F must be a bare function pointer"
        );
        // SAFETY: delegated to caller per the doc-comment; we only reinterpret
        // a `*const u8` as a function pointer with matching layout.
        unsafe {
            let p: *const u8 = self.code.cast_const();
            std::mem::transmute_copy::<*const u8, F>(&p)
        }
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: `code` / `size` came from `mmap` in `load`.
        unsafe {
            libc::munmap(self.code.cast::<c_void>(), self.size);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is thread-safe and always defined for `_SC_PAGESIZE`.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(p > 0, "sysconf(_SC_PAGESIZE) failed");
    usize::try_from(p).expect("page size fits in usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_and_execute_simple_function() {
        // `mov rax, 42; ret` is the minimal SysV int-return function.
        // Encoding: B8 2A 00 00 00 00 00 00 00 (no — that's mov eax, 42). For
        // 64-bit: 48 C7 C0 2A 00 00 00  C3
        let code: &[u8] = &[
            0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00, // mov rax, 42
            0xC3, // ret
        ];
        let m = Module::load(code).expect("mmap+mprotect succeeds");
        type Fn0 = unsafe extern "sysv64" fn() -> i64;
        // SAFETY: the emitted code has the exact signature of Fn0.
        let f: Fn0 = unsafe { m.entry() };
        let r = unsafe { f() };
        assert_eq!(r, 42);
    }

    #[test]
    fn load_rejects_empty_buffer() {
        let result = std::panic::catch_unwind(|| Module::load(&[]));
        assert!(result.is_err(), "loading empty bytes must panic");
    }

    #[test]
    fn module_reports_its_mapping_size_rounded_to_page() {
        let code: &[u8] = &[0xC3]; // ret
        let m = Module::load(code).unwrap();
        let page = page_size();
        assert!(m.size() >= page);
        assert_eq!(m.size() % page, 0);
    }
}
