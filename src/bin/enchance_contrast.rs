use histeq::{Method, transform_any_8bit_image, transform_png_image};

fn main() {
    let self_name = std::env::args().nth(0).unwrap();
    let unsage_str = format!("Usage:\n{} filename_in filename_out method\nwhere method is one of HE, CLAHE", self_name);
    let filename_in = std::env::args().nth(1).expect(&format!("{}\n no filename_in found", unsage_str));
    let filename_out = std::env::args().nth(2).expect(&format!("{}\n no filename_out found", unsage_str));
    let method_name = std::env::args().nth(3).expect(&format!("{}\n no method found", unsage_str));
    let method: Method = match method_name.to_lowercase().as_str() {
        "he" => Method::HE,
        "clahe" => Method::CLAHE,
        "he_hsl" => Method::HE_HSL,
        "clahe_hsl" => Method::CLAHE_HSL,
        v => panic!("Wrong method: {}", v)
    };

    if filename_in.to_lowercase().ends_with("png") {
        transform_png_image(&filename_in, &filename_out, 8, method);
    } else {
        transform_any_8bit_image(&filename_in, &filename_out, 8, method);
    }
}