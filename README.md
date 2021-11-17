# histeq
CLAHE and AHE methods implemented in Rust with ndarray.

Reimplementation of https://github.com/inferrna/histogram_equalization so you can use as example of migrating from python+numpy.<br>
(Currently rust version is about 2 times faster, so it worth that.)<br>
Also look at https://docs.rs/ndarray/0.15.3/ndarray/doc/ndarray_for_numpy_users/index.html

## Supported methods
  * HE
  * CLAHE

## Usage
```bash
$ cargo build --release
$ ./target/release/enchance_contrast input.png oputput.png CLAHE
```
**Supports 16-bit png images**
