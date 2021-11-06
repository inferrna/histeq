use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Deref;
use image::{DynamicImage, GenericImageView, GrayImage, RgbImage};
use lodepng::{Bitmap, ColorType, Image, RGB};
use ndarray::{s, Array0, Array1, Array2, Array3, ArrayView2, Axis, Array};
use ndarray_stats::QuantileExt;
use itertools::Itertools;
use num_traits::{FromPrimitive, One, PrimInt, sign::Unsigned, ToPrimitive};
use image::io::Reader as ImageReader;

fn load_16bit_png_as_array(filename: &str) -> (usize, Array3<u16>) { //Level, data
    let image: Image = lodepng::decode_file(filename, ColorType::RGB, 16).unwrap();
    let bm = match image {
        Image::RGB16(b) => {b},
        v => {panic!("Supposed to be 16 bit RGB, got {:?}", v)}
    };
    //image as Bitmap<RGB<u16>>;
    //let buf = bm.buffer.iter().map(|v| vec![v.r as f32, v.g as f32, v.b as f32]).flatten().collect::<Vec<f32>>();
    let buf = bm.buffer.iter().map(|v| vec![v.r, v.g, v.b]).flatten().collect::<Vec<u16>>();
    (65536, Array3::from_shape_vec((bm.height, bm.width, 3), buf).unwrap())
}

fn load_8bit_img_as_array(filename: &str) -> (usize, Array3<u8>) { //Level, data
    let image = ImageReader::open(filename).unwrap().decode().unwrap();
    let (h, w) = (image.height() as usize, image.width() as usize);
    let (bm, cc) = match image {
        DynamicImage::ImageLuma8(i) => {(i.into_vec(), 1)},
        DynamicImage::ImageRgb8(i) => {(i.into_vec(), 3)},
        DynamicImage::ImageBgr8(i) => {(i.into_vec(), 3)},
        v => {panic!("Supposed to be 8 bit image, got {:?}", v)}
    };
    (256, Array3::from_shape_vec((h, w, cc), bm).unwrap())
}

/*
fn clahe_filename_16bit_png(filename: &str) -> Array2<u16> {
    let (level, array) = load_16bit_png_as_array(filename);
    //let brightness: Array2<f32> = array.map_axis(Axis(2), |v|*v.iter().max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap());
    let brightness: Array2<u16> = array.map_axis(Axis(2), |v|*v.iter().max().unwrap());
    let result: Array2<f32> = clahe_2d(&brightness, level, 8);
    brightness.mu
    result
}
*/

fn clahe_filename_8bit_image(filename: &str) {
    let (level, array) = load_8bit_img_as_array(filename);
    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let cc = shape[2];
    //let brightness: Array2<f32> = array.map_axis(Axis(2), |v|*v.iter().max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal)).unwrap());
    let brightness: Array2<u8> = array.map_axis(Axis(2), |v|*v.iter().max().unwrap());
    dbg!(level);
    let tuned_brightness: Array2<f32> = clahe_2d(&brightness, level, 8);
    let tuned_brightness = tuned_brightness.into_shape((h as usize, w as usize, 1)).unwrap();
    let brightness = brightness.into_shape((h as usize, w as usize, 1)).unwrap();
    dbg!(brightness.shape());
    dbg!(tuned_brightness.shape());
    dbg!(array.shape());
    let result_float: Array3<f32> = array.mapv(|elem| elem as f32) * tuned_brightness / brightness.mapv(|elem| elem as f32);
    let result_bytes: Array3<u8> = result_float.mapv(|elem| elem.round() as u8);

    let img: DynamicImage = match cc {
        3 => DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        1 => DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        _ => unimplemented!()
    };
    img.save("/tmp/clahe_rust_result.png");
}

fn clahe_2d<I>(img_array: &Array2<I>, level: usize, blocks: usize) -> Array2<f32>
where I: PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug
{
    dbg!(level);
    let (m, n) = if let [mm, nn] = img_array.shape().clone() {
        (*mm as usize, *nn as usize)
    } else {
        panic!("Supposed to 2D array")
    };
    let block_m = 2 * (m as f32 / (2 * blocks) as f32).ceil() as usize;
    let block_n = 2 * (n as f32 / (2 * blocks) as f32).ceil() as usize;

    let mut maps: Vec<Vec<Array1<I>>> = vec![vec![]; blocks];

    for i in 0..blocks {
        for j in 0..blocks {
            //block border
            let (si, ei) = (i * block_m, (i + 1) * block_m);
            let (sj, ej) = (j * block_n, (j + 1) * block_n);

            let block_view = img_array.slice(s![si..ei.min(m), sj..ej.min(n)]);
            let hist = calc_hist(&block_view, level);
            let hist_cdf = calc_hist_cdf(&hist, level);
            maps[i].push(hist_cdf);
        }
    }


    let block_m = block_m as isize;
    let block_n = block_n as isize;

    let block_m_step = (block_m / 2);
    let block_n_step = (block_n / 2);

    let mut array_result: Array2<f32> = Array2::zeros((m as usize, n as usize,));

    let iblocks = blocks as isize;

    for m_start in (0..m as isize).step_by(block_m_step as usize) {
        for n_start in (0..n as isize).step_by(block_n_step as usize) {
            let range_i = (m_start..(m_start + block_m_step).min(m as isize));
            let range_j = (n_start..(n_start + block_n_step).min(n as isize));
            let arr_i = Array1::from_iter(&mut range_i.clone());
            let arr_j = Array1::from_iter(&mut range_j.clone());

            //arr_r: np.ndarray = np.floor((arr_i.astype(np.float32) - block_m_step) / block_m).astype(np.int)
            //arr_c: np.ndarray = np.floor((arr_j.astype(np.float32) - block_n_step) / block_n).astype(np.int)
            let arr_r = (&arr_i - block_m_step) / block_m;
            let arr_c = (&arr_j - block_n_step) / block_n;

            let r = arr_r[0];
            let c = arr_c[0];

            // arr_x1: np.ndarray = (
            //     (arr_i.astype(np.float32) - (arr_r.astype(np.float32) + 0.5) * block_m) / block_m).astype(
            //     np.float32)
            // arr_y1: np.ndarray = (
            //     (arr_j.astype(np.float32) - (arr_c.astype(np.float32) + 0.5) * block_n) / block_n).astype(
            //     np.float32).reshape(-1, 1)

            let arr_x1: Array1<f32> = arr_i.mapv(|elem| elem as f32 / block_m as f32) - arr_r.mapv(|elem| elem as f32) - 0.5;
            let arr_y1: Array1<f32> = arr_j.mapv(|elem| elem as f32 / block_n as f32) - arr_c.mapv(|elem| elem as f32) - 0.5;

            let arr_x1_sub: Array1<f32> = (1.0 - &arr_x1);
            let arr_y1_sub: Array1<f32> = (1.0 - &arr_y1);

            let mut new_x_shape = (arr_x1.shape()[0], 1);

            let arr_x1 = arr_x1.into_shape(new_x_shape.clone()).unwrap();
            let arr_x1_sub = arr_x1_sub.into_shape(new_x_shape).unwrap();


            if r < iblocks-1 && c < iblocks-1 {
                let rl = r as usize;
                let cl = c as usize;

                dbg!(arr_y1_sub.shape());
                dbg!(arr_x1_sub.shape());
                dbg!(arr_x1.shape());
                dbg!(arr_y1.shape());

                let img_arr_idx = img_array.slice(s![range_i.clone(), range_j.clone()]);
                let mapped_lu = img_arr_idx.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()].to_f32().unwrap());
                let mapped_lb = img_arr_idx.mapv(|elem| maps[rl + 1][cl][elem.to_usize().unwrap()].to_f32().unwrap());
                let mapped_ru = img_arr_idx.mapv(|elem| maps[rl][cl + 1][elem.to_usize().unwrap()].to_f32().unwrap());
                let mapped_rb = img_arr_idx.mapv(|elem| maps[rl + 1][cl + 1][elem.to_usize().unwrap()].to_f32().unwrap());

                dbg!(mapped_lu.shape());
                dbg!(mapped_lb.shape());
                dbg!(mapped_ru.shape());
                dbg!(mapped_rb.shape());

                let xs_mlu = &arr_x1_sub * mapped_lu;
                let x_mlb = &arr_x1 * mapped_lb;

                let xs_mru = &arr_x1_sub * mapped_ru;
                let x_mrb = &arr_x1 * mapped_rb;

                let mut mapped_mult_sum: Array2<f32> = arr_y1_sub * (xs_mlu + x_mlb) + arr_y1 * (xs_mru + x_mrb);
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            }
        }
    }
    array_result//.mapv(|elem| I::from_f32(elem.round()).unwrap())
}

fn calc_hist<I>(img_array: &ArrayView2<I>, level: usize) -> Array1<usize>
where I: PrimInt + Unsigned + ToPrimitive + std::ops::AddAssign + One + Debug
{
    dbg!(level);
    let mut hist: Array1<usize> = Array1::zeros((level,));
    img_array.for_each(|v| hist[v.to_usize().unwrap()] += 1);
    hist
}

fn calc_hist_cdf<I>(hist: &Array1<usize>, level: usize) -> Array1<I>
where I: PrimInt + Unsigned + FromPrimitive + std::ops::AddAssign + One + Debug
{
    let first_nz = *hist.iter().enumerate().find_or_first(|(v, i)| v>&0).unwrap().1.max(&1);
    let mut hist_cumsum: Array1<usize> = hist.mapv(|elem| elem.to_usize().unwrap());
    let length = hist.len();
    dbg!(length);
    for i in 1..length {
        hist_cumsum[i] += hist_cumsum[i-1]
    }
    dbg!(&hist_cumsum);

    let mut hist_cumsum = hist_cumsum.mapv(|elem| elem as f32);

    let avg_level = (level - 1);

    let const_a = avg_level as f32 / *hist_cumsum.max().unwrap() as f32;

    let cf = (avg_level - first_nz) as f32 / avg_level as f32;

    hist_cumsum *= (const_a * cf);

    hist_cumsum += first_nz as f32;
    hist_cumsum.mapv(|elem| I::from_f32( elem.round()).unwrap())
}

#[cfg(test)]
mod tests {
    use crate::clahe_filename_8bit_image;

    #[test]
    fn test_8_bit() {
        clahe_filename_8bit_image("car.png");
    }
}
