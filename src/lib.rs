use std::cmp::Ordering;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufWriter, Cursor};
use std::ops::Deref;
use std::path::Path;
use byteorder::BigEndian;
use image::{DynamicImage, GenericImageView, GrayImage, RgbImage};
use byteorder::ReadBytesExt;
use png::BitDepth;
use ndarray::{s, Array0, Array1, Array2, Array3, ArrayView2, Axis, Array};
use ndarray_stats::QuantileExt;
use itertools::{Itertools, Merge};
use num_traits::{Bounded, FromPrimitive, One, PrimInt, sign::Unsigned, ToPrimitive};
use image::io::Reader as ImageReader;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub enum Method {
    HE,
    CLAHE
}

pub fn transform_png_image(filename_in: &str, filename_out: &str, blocks: usize, method: Method) { //Level, data
    let img_file = File::open(filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    // Use the IDENTITY transformation because by default
    // it will use STRIP_16 which only keep 8 bits.
    decoder.set_transformations(png::Transformations::IDENTITY);
    let (info, mut reader) = decoder.read_info().unwrap();

    let path_out = Path::new(filename_out);
    let file = File::create(path_out).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, info.width, info.height); // Width is 2 pixels and height is 1.
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);

    match info.bit_depth {
        BitDepth::Eight => {
            let mut buffer = vec![0; info.buffer_size()];
            reader.next_frame(&mut buffer).unwrap();
            let array_image = Array3::from_shape_vec((info.height as usize, info.width as usize, info.color_type.samples()), buffer).unwrap();
            let result_bytes = equalize_full_image(&array_image, blocks, method);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&result_bytes.into_iter().collect::<Vec<u8>>());
        }
        BitDepth::Sixteen => {
            let mut buffer = vec![0; info.buffer_size()];
            reader.next_frame(&mut buffer).unwrap();
            let (h, w, cc) = (info.height as usize, info.width as usize, info.color_type.samples());
            let mut buffer_u16 = vec![0; h * w * cc];
            let mut buffer_cursor = Cursor::new(buffer);
            buffer_cursor
                .read_u16_into::<BigEndian>(&mut buffer_u16)
                .unwrap();
            let array_image = Array3::from_shape_vec((h, w, cc), buffer_u16).unwrap();
            let result_bytes = equalize_full_image(&array_image, blocks, method);

            let mut buffer_u8_le = result_bytes.into_iter().map(|v: u16| v.to_be_bytes()).flatten().collect::<Vec<u8>>();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&buffer_u8_le);

        }
        v => {panic!("Supposed to be 8 or 16 bit image, got {:?}", v)}
    }
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


pub fn transform_any_8bit_image(filename_in: &str, filename_out: &str, blocks: usize, method: Method) {
    let (level, array) = load_8bit_img_as_array(filename_in);

    let shape = array.shape();
    let h = shape[0] as u32;
    let w = shape[1] as u32;
    let cc = shape[2];

    let result_bytes = equalize_full_image(&array, blocks, method);

    let img: DynamicImage = match cc {
        3 => DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        1 => DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        _ => unimplemented!()
    };
    img.save(filename_out);
}

fn equalize_full_image<I>(array: &Array3<I>, blocks: usize, method: Method) -> Array3<I>
where I: PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded, u32: From<I>, f32: From<I>, usize: From<I>, u32: From<I>
{
    let max_val = u32::from(I::max_value());
    let (h, w, cc) = if let [hh, ww, ccc] = array.shape().clone() {
        (*hh, *ww, *ccc)
    } else {
        panic!("Supposed to be 3D array")
    };
    // Brightness level as max value of RGB
    let brightness = if cc > 1 {
        array.map_axis(Axis(2), |v|*v.iter().max().unwrap()).clone()
    } else {
        array.map_axis(Axis(2), |v|v[0]).clone()
    };
    let tuned_brightness: Array2<f32> = match method {
        Method::HE => {he_2d(&brightness)}
        Method::CLAHE => {clahe_2d(&brightness, blocks)}
    };
    let tuned_brightness = tuned_brightness.into_shape((h as usize, w as usize, 1)).unwrap();
    let brightness = brightness.into_shape((h as usize, w as usize, 1)).unwrap();

    // Align full RGB image
    let result_float: Array3<f32> = array.mapv(|elem| f32::from(elem)) * tuned_brightness / brightness.mapv(|elem| f32::from(elem.max(I::one())));
    let result_bytes: Array3<I> = result_float.mapv(|elem| I::from_u32((elem.round() as u32).min(max_val)).unwrap());

    result_bytes
}

fn he_2d<I>(img_array: &Array2<I>) -> Array2<f32>
where I: PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded, usize: From<I>, f32: From<I>
{
    let mut hist = calc_hist(&img_array.view());
    clip_hist(&mut hist, 10.0);
    let hist_cdf: Array1<f32> = calc_hist_cdf(&hist);

    let result = img_array.mapv(|v: I| hist_cdf[usize::from(v)]);
    result
}

fn clahe_2d<I>(img_array: &Array2<I>, blocks: usize) -> Array2<f32>
where I: PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded, usize: From<I>
{
    let (m, n) = if let [mm, nn] = img_array.shape().clone() {
        (*mm as usize, *nn as usize)
    } else {
        panic!("Supposed to be 2D array")
    };
    let block_m = 2 * (m as f32 / (2 * blocks) as f32).ceil() as usize;
    let block_n = 2 * (n as f32 / (2 * blocks) as f32).ceil() as usize;

    let mut maps: Vec<Vec<Array1<f32>>> = vec![vec![]; blocks];

    for i in 0..blocks {
        for j in 0..blocks {
            //block border
            let (si, ei) = (i * block_m, (i + 1) * block_m);
            let (sj, ej) = (j * block_n, (j + 1) * block_n);

            let block_view = img_array.slice(s![si..ei.min(m), sj..ej.min(n)]);
            let mut hist = calc_hist(&block_view);
            clip_hist(&mut hist, 10.0);
            let hist_cdf = calc_hist_cdf(&hist);
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

            let arr_r = (&arr_i - block_m_step) / block_m;
            let arr_c = (&arr_j - block_n_step) / block_n;

            let r = arr_r[0];
            let c = arr_c[0];

            let arr_x1: Array1<f32> = arr_i.mapv(|elem| elem as f32 / block_m as f32) - arr_r.mapv(|elem| elem as f32) - 0.5;
            let arr_y1: Array1<f32> = arr_j.mapv(|elem| elem as f32 / block_n as f32) - arr_c.mapv(|elem| elem as f32) - 0.5;

            let arr_x1_sub: Array1<f32> = (1.0 - &arr_x1);
            let arr_y1_sub: Array1<f32> = (1.0 - &arr_y1);

            let mut new_x_shape = (arr_x1.shape()[0], 1);

            let arr_x1 = arr_x1.into_shape(new_x_shape.clone()).unwrap();
            let arr_x1_sub = arr_x1_sub.into_shape(new_x_shape).unwrap();

            let img_arr_idx = img_array.slice(s![range_i.clone(), range_j.clone()]);

            let corner_block = if r < 0 && c < 0 {
                Some(((r+1) as usize, (c+1) as usize))
            } else if r < 0 && c >=iblocks-1 {
                Some(((r+1) as usize, c as usize))
            } else if r > 0 && c < 0 {
                Some((r as usize, (c+1) as usize))
            } else if r >=iblocks-1 && c >=iblocks-1 {
                Some ((r as usize, c as usize))
            } else {
                None
            };

            if let Some((rl, cl)) = corner_block {
                let img_arr_idx = img_array.slice(s![range_i.clone(), range_j.clone()]);
                let mapped = img_arr_idx.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                mapped.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if (r < iblocks-1 && c < iblocks-1) && (r >=0 && c >=0) {
                let rl = r as usize;
                let cl = c as usize;

                let mapped_lu = img_arr_idx.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_lb = img_arr_idx.mapv(|elem| maps[rl + 1][cl][elem.to_usize().unwrap()]);
                let mapped_ru = img_arr_idx.mapv(|elem| maps[rl][cl + 1][elem.to_usize().unwrap()]);
                let mapped_rb = img_arr_idx.mapv(|elem| maps[rl + 1][cl + 1][elem.to_usize().unwrap()]);

                let xs_mlu = &arr_x1_sub * mapped_lu;
                let x_mlb = &arr_x1 * mapped_lb;

                let xs_mru = &arr_x1_sub * mapped_ru;
                let x_mrb = &arr_x1 * mapped_rb;

                let mut mapped_mult_sum: Array2<f32> = arr_y1_sub * (xs_mlu + x_mlb) + arr_y1 * (xs_mru + x_mrb);
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if r < 0 || r >=iblocks-1 {
                let rl = r.max(0).min(iblocks-1) as usize;
                let cl = c as usize;
                let mapped_left = arr_y1_sub * img_arr_idx.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_right = arr_y1 * img_arr_idx.mapv(|elem| maps[rl][cl+1][elem.to_usize().unwrap()]);
                let mapped_mult_sum = mapped_left + mapped_right;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if c < 0 || c >=iblocks-1 {
                let rl = r as usize;
                let cl = c.max(0).min(iblocks-1) as usize;
                let mapped_up = arr_x1_sub * img_arr_idx.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_bottom = arr_x1 * img_arr_idx.mapv(|elem| maps[rl+1][cl][elem.to_usize().unwrap()]);
                let mapped_mult_sum = mapped_up + mapped_bottom;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else {
                panic!("Should not be reached! r={}, c={}, iblocks={}", r, c, iblocks)
            }
        }
    }
    array_result
}


fn clip_hist(hist: &mut Array1<f32>, threshold: f32) {
    let all_sum = hist.sum() as f32;
    let threshold_value = (threshold * all_sum / hist.len() as f32);
    let total_extra: f32 = hist.iter().filter(|v| v >= &&threshold_value).map(|v| v - threshold_value).sum();
    let mean_extra = (total_extra / hist.len() as f32);

    hist.map_mut(|v: &mut f32| if *v >= threshold_value {*v = threshold_value + mean_extra} else {*v += mean_extra});
}

fn calc_hist<I>(img_array: &ArrayView2<I>) -> Array1<f32>
where I: PrimInt + Unsigned + ToPrimitive + std::ops::AddAssign + One + Debug + Bounded, usize: From<I>
{
    let level = usize::from(I::max_value())+1;
    let mut hist: Array1<f32> = Array1::zeros((level,));
    img_array.for_each(|v| hist[v.to_usize().unwrap()] += 1.0);
    hist
}

fn calc_hist_cdf<I>(hist: &Array1<f32>) -> Array1<f32>
where I: PrimInt + Unsigned + FromPrimitive + std::ops::AddAssign + One + Debug + Bounded, usize: From<I>
{
    let level = usize::from(I::max_value())+1;
    let first_nz = hist.iter().enumerate().find_or_first(|(i, v)| v>&&3.0).unwrap().0.max(1);
    let last_nz = hist.iter().enumerate().rev().find_or_first(|(i, v)| v>&&3.0).unwrap().0.max(1);
    let mut hist_cumsum: Array1<f32> = hist.mapv(|elem| elem);
    let length = hist.len();
    //dbg!(length);
    for i in 1..length {
        hist_cumsum[i] += hist_cumsum[i-1]
    }

    //Limit contrast range to near original
    //let max_level = (last_nz + level - 1) / 2;
    //let min_level = first_nz / 2;

    // Unlimited contrast
    let max_level = level - 1;
    let min_level = 0;

    let const_a = max_level as f32 / *hist_cumsum.max().unwrap();

    let cf = (max_level - min_level) as f32 / max_level as f32;

    hist_cumsum *= (const_a * cf);

    hist_cumsum += min_level as f32;
    hist_cumsum
}

#[cfg(test)]
mod tests {
    use crate::{Method, transform_any_8bit_image};

    #[test]
    fn test_8_bit() {
        transform_any_8bit_image("car.png", "car_out.png", 8, Method::CLAHE);
    }
}
