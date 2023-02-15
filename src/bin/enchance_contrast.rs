use histeq::{Method, transform_any_8bit_image, transform_png_image};

fn main() {
    let self_name = std::env::args().next().unwrap();
    let usage_str = format!("Usage:\n{self_name} filename_in filename_out method\nwhere method is one of HE, CLAHE");
    let filename_in = std::env::args().nth(1).unwrap_or_else(|| panic!("{usage_str}\n no filename_in found"));
    let filename_out = std::env::args().nth(2).unwrap_or_else(|| panic!("{usage_str}\n no filename_out found"));
    let method_name = std::env::args().nth(3).unwrap_or_else(|| panic!("{usage_str}\n no method found"));
    let method: Method = match method_name.to_lowercase().as_str() {
        "he" => Method::HE,
        "clahe" => Method::CLAHE,
        "he_hsl" => Method::HE_HSL,
        "clahe_hsl" => Method::CLAHE_HSL,
        "he_noisy" => Method::HE_NOISY,
        "clahe_noisy" => Method::CLAHE_NOISY,
        v => panic!("Wrong method: {v}")
    };

    if filename_in.to_lowercase().ends_with("png") {
        transform_png_image(&filename_in, &filename_out, 8, method);
    } else {
        transform_any_8bit_image(&filename_in, &filename_out, 8, method);
    }
}