use image::{Rgb, RgbImage};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Parameters {
    ///Path to the input file.
    #[arg(long)]
    height: u32,
    ///Output file path
    #[arg(long)]
    width: u32,
}

fn main() {
    let params = Parameters::parse();
    let mut green_length = 0.0;
    let mut step = 1.0 / params.height as f32;

    let mut img_out = RgbImage::new(params.width, params.height);

    for y in 0..params.height {
        let r_brightness= 15 + (y as f32 * step * 80 as f32).round() as u8;
        let g_brightness= 95 + ((1.0 - y as f32 * step) * 80 as f32).round() as u8;
        let b_brightness= 175 + (y as f32 * step * 80 as f32).round() as u8;

        green_length = y as f32 * step * params.width as f32;
        let gl_i = green_length.round() as u32;

        let rbl_i = (params.width - gl_i) / 2;

        for x in 0..rbl_i { //Red triangle
            img_out.put_pixel(x, y, Rgb::from([r_brightness, 0, 0]))
        }
        for x in rbl_i..rbl_i+gl_i { //Green triangle
            img_out.put_pixel(x, y, Rgb::from([0, g_brightness, 0]))
        }
        for x in rbl_i+gl_i..params.width { //Blue triangle
            img_out.put_pixel(x, y, Rgb::from([0, 0, b_brightness]))
        }
    }
    img_out.save("/tmp/test_triangles.png").unwrap();
}