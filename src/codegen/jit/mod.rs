//! Tiny JIT runtime: mmap a RW region, memcpy the code, flip to RX, hand out
//! function pointers.
//!
//! Linux/macOS (POSIX) only. Windows would need `VirtualAlloc` /
//! `VirtualProtect` — punted until we need it.

use std::ffi::{CString, c_void};
use std::io;
use std::ptr;

/// A JIT-loaded code region. `Drop` munmaps.
pub struct Module {
    code: *mut u8,
    size: usize,
}

/// One symbol-patch request: a byte offset into the code blob where an
/// 8-byte function-pointer immediate lives, plus the symbol's name.
#[derive(Clone, Debug)]
pub struct Relocation {
    pub offset: usize,
    pub symbol: String,
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
        Self::load_with_relocs(bytes, &[], "")
    }

    /// Load code and resolve call-site relocations. Each relocation is
    /// an 8-byte slot in the code at `offset`; the resolver looks up
    /// the symbol (either `self_symbol` → the module's own base or
    /// `dlsym(RTLD_DEFAULT, name)`) and writes the address in place.
    ///
    /// # Errors
    /// `mmap`/`mprotect` errors, or `ErrorKind::NotFound` if a symbol
    /// can't be resolved.
    pub fn load_with_relocs(
        bytes: &[u8],
        relocations: &[Relocation],
        self_symbol: &str,
    ) -> io::Result<Self> {
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

        // Apply relocations: look each symbol up (either the module's
        // own name → base address, or via dlsym) and write an 8-byte
        // LE pointer at the recorded offset.
        for reloc in relocations {
            let addr = if !self_symbol.is_empty() && reloc.symbol == self_symbol {
                ptr.cast::<u8>() as u64
            } else if let Some(a) = resolve_external(&reloc.symbol) {
                a
            } else {
                // SAFETY: we own the mapping.
                unsafe {
                    libc::munmap(ptr, size);
                }
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("unresolved symbol: {}", reloc.symbol),
                ));
            };
            // SAFETY: `reloc.offset + 8 <= bytes.len() <= size`; both
            // bytes and ptr live for `size`.
            assert!(
                reloc.offset + 8 <= bytes.len(),
                "relocation offset {} overflows code buffer len {}",
                reloc.offset,
                bytes.len()
            );
            unsafe {
                let slot_bytes = addr.to_le_bytes();
                let dst = ptr.cast::<u8>().add(reloc.offset);
                ptr::copy_nonoverlapping(slot_bytes.as_ptr(), dst, 8);
            }
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

/// Resolve an external symbol to an absolute address via `dlsym`.
/// Returns `None` if the symbol isn't in any loaded library.
fn resolve_external(symbol: &str) -> Option<u64> {
    ensure_extra_libs_loaded();
    let cname = CString::new(symbol).ok()?;
    // SAFETY: `RTLD_DEFAULT` is a portable sentinel; `dlsym` on a
    // nul-terminated C string is defined. We treat the returned
    // pointer as an opaque address and never dereference it here.
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, cname.as_ptr()) };
    if addr.is_null() {
        None
    } else {
        Some(addr as u64)
    }
}

/// Process-wide one-shot to pull libraries commonly referenced by
/// JITed LLVM IR into our address space so `dlsym(RTLD_DEFAULT, ...)`
/// can find them. Rust binaries don't link libstdc++ by default, so
/// `_Znwm`, `_ZdlPvm`, and friends are unresolvable without this.
fn ensure_extra_libs_loaded() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for libname in [b"libstdc++.so.6\0".as_ref(), b"libm.so.6\0".as_ref()] {
            // SAFETY: Both candidates are nul-terminated C strings;
            // `dlopen` returns NULL on failure which we tolerate
            // (the platform may genuinely lack one of these).
            unsafe {
                let _ = libc::dlopen(
                    libname.as_ptr().cast::<std::os::raw::c_char>(),
                    libc::RTLD_NOW | libc::RTLD_GLOBAL,
                );
            }
        }
    });
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
