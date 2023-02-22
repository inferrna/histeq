use histeq::{BlocksCount, BrightnessLimits, HEParams, Method, transform_any_8bit_image, transform_png_image};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Parameters {
    ///Path to the input file.
    #[arg(long)]
    filename_in: String,
    ///Output file path
    #[arg(long)]
    filename_out: String,
    ///Equalization method
    #[arg(long)]
    method: Method,
    ///Blocks count by axis Y. Default to 8
    #[arg(long)]
    blocks_h: Option<usize>,
    #[arg(long)]
    ///Blocks count by axis X. Optional, can be calculated from blocks_h
    blocks_w: Option<usize>,
    #[arg(long)]
    ///Low value for the new histogram. 0.0 means zero, 1.0 means darkest level of original histogram. Default to 0.5.
    dark_limit: Option<f32>,
    #[arg(long)]
    ///High value for the new histogram. 0.0 means brightest level of original histogram, 1.0 means max possible value. Default to 0.5.
    bright_limit: Option<f32>,
    #[cfg(feature = "denoise")]
    #[arg(long)]
    ///Use denoise for histogram computation. Default to false
    use_denoise: bool
}

fn main() {
    let params = Parameters::parse();

    let blocks: BlocksCount = BlocksCount::new(params.blocks_h.unwrap_or(8), params.blocks_w);
    let limits: BrightnessLimits = BrightnessLimits::new(params.dark_limit.unwrap_or(0.5), params.bright_limit.unwrap_or(0.5));

    #[cfg(feature = "denoise")]
    let denoise = params.use_denoise;
    #[cfg(not(feature = "denoise"))]
    let denoise = false;

    let he_params = HEParams::new(blocks, limits, denoise);

    if params.filename_in.to_lowercase().ends_with("png") {
        transform_png_image(&params.filename_in, &params.filename_out, params.method, &he_params);
    } else {
        transform_any_8bit_image(&params.filename_in, &params.filename_out, params.method, &he_params);
    }
}
