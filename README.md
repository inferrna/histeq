# histeq
CLAHE and HE methods implemented in Rust with ndarray.

Rust reimplementation of https://github.com/inferrna/histogram_equalization .<br>
(Currently rust version is about 2 times faster.)<br>
Also look at https://docs.rs/ndarray/0.15.3/ndarray/doc/ndarray_for_numpy_users/index.html

## Supported methods
  * HE
  * CLAHE

## Usage
```bash
$ cargo build --release --bin enchance_contrast --features "denoise"
Usage: enchance_contrast [OPTIONS] --filename-in <FILENAME_IN> --filename-out <FILENAME_OUT> --method <METHOD>

Options:
      --filename-in <FILENAME_IN>
          Path to the input file

      --filename-out <FILENAME_OUT>
          Output file path

      --method <METHOD>
          Equalization method

          Possible values:
          - he-hsl:      Histogram equalization across all image in HSL color space
          - clahe-hsl:   Contrast limited histogram equalization in HSL color space
          - he:          Histogram equalization across all image
          - clahe:       Contrast limited histogram equalization
          - he-noisy:    Histogram equalization across all image with no respect to noisy pixels
          - clahe-noisy: Contrast limited histogram equalization with no respect to noisy pixels

      --blocks-h <BLOCKS_H>
          Blocks count by axis Y. Default to 8

      --blocks-w <BLOCKS_W>
          Blocks count by axis X. Optional, can be calculated from blocks_h

      --dark-limit <DARK_LIMIT>
          Low value for the new histogram. 0.0 means zero, 1.0 means darkest level of original histogram. Default to 0.5

      --bright-limit <BRIGHT_LIMIT>
          High value for the new histogram. 0.0 means brightest level of original histogram, 1.0 means max possible value. Default to 0.5

      --use-denoise
          Use denoise for histogram computation. Default to false

  -h, --help
          Print help (see a summary with '-h')

  -V, --version

```
**Supports 16-bit png images**
