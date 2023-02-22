mod line_up_colors;
mod he_clahe_hsl;
mod plot_histogram;


use std::fmt::Debug;
use std::fs::File;
use std::io::{BufWriter, Cursor};

use std::path::Path;
use std::rc::Rc;
use byteorder::BigEndian;
use image::{DynamicImage, GrayImage, RgbImage};
use byteorder::ReadBytesExt;
use png::BitDepth;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s, Zip};
use ndarray_stats::QuantileExt;

use num_traits::{Bounded, FromPrimitive, Num, PrimInt, sign::Unsigned, ToPrimitive};
use image::io::Reader as ImageReader;
use crate::he_clahe_hsl::{clahe_2d_hsl, he_2d_hsl};
use crate::line_up_colors::HSLable;
use clap::{Parser, ValueEnum};
#[cfg(feature = "denoise")]
use smart_denoise::{Denoiseable, Algo, DenoiseParams, UsingShader};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ValueEnum, Parser)]
pub enum Method {
    ///Histogram equalization across all image in HSL color space
    HE_HSL,
    ///Contrast limited histogram equalization in HSL color space
    CLAHE_HSL,
    ///Histogram equalization across all image
    HE,
    ///Contrast limited histogram equalization
    CLAHE,
    ///Histogram equalization across all image with no respect to noisy pixels
    HE_NOISY,
    ///Contrast limited histogram equalization with no respect to noisy pixels
    CLAHE_NOISY,
}

#[derive(Debug, Clone)]
pub struct BlocksCount {
    size_h: usize,
    size_w: Option<usize>
}

#[derive(Debug, Clone)]
pub struct BrightnessLimits {
    dark_limit: f32,
    bright_limit: f32
}

impl BrightnessLimits {
    pub fn new(dark_limit: f32, bright_limit: f32) -> Self {
        Self { dark_limit, bright_limit }
    }
}

impl BlocksCount {
    pub fn new(size_h: usize, size_w: Option<usize>) -> Self {
        Self { size_h, size_w }
    }
}

#[derive(Debug, Clone)]
pub struct HEParams {
    blocks: BlocksCount,
    limits: BrightnessLimits,
    denoise: bool,
}

impl HEParams {
    pub fn new(blocks: BlocksCount, limits: BrightnessLimits, denoise: bool) -> Self {
        Self { blocks, limits, denoise }
    }
    pub fn blocks(&self) -> &BlocksCount {
        &self.blocks
    }
    pub fn limits(&self) -> &BrightnessLimits {
        &self.limits
    }
    pub fn denoise(&self) -> bool {
        self.denoise
    }
}

#[cfg(feature = "denoise")]
///Returns original image and denoise variant if denoise was requested. Two copy of original image otherwise.
fn image_pair_from_buffer<I: Denoiseable>(buffer: Vec<I>, h: usize, w: usize, cc: usize, denoise: bool) -> (Rc<Array3<I>>, Rc<Array3<I>>) {
    let denoise_params = DenoiseParams::new(3.0, 2.0, 0.185);
    if denoise {
        let dnd_buf = smart_denoise::denoise(&buffer, w as u32, h as u32, UsingShader::Compute, denoise_params, false, Algo::Smart);
        let array_image = Array3::from_shape_vec((h, w, cc), buffer).unwrap();
        let array2hist = Array3::from_shape_vec((h, w, cc), dnd_buf).unwrap();
        (Rc::new(array_image), Rc::new(array2hist))
    } else {
        let array_image = Rc::new(Array3::from_shape_vec((h, w, cc), buffer).unwrap());
        (array_image.clone(), array_image)
    }
}
#[cfg(not(feature = "denoise"))]
///Returns original image and copy of original image.
fn image_pair_from_buffer<I: Num>(buffer: Vec<I>, h: usize, w: usize, cc: usize, denoise: bool) -> (Rc<Array3<I>>, Rc<Array3<I>>)    {
    let array_image = Rc::new(Array3::from_shape_vec((h, w, cc), buffer).unwrap());
    (array_image.clone(), array_image)
}


pub fn transform_png_image(filename_in: &str, filename_out: &str, method: Method, params: &HEParams) { //Level, data
    let img_file = File::open(filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    // Use the IDENTITY transformation because by default
    // it will use STRIP_16 which only keep 8 bits.
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut png_reader = decoder.read_info().unwrap();
    let info = png_reader.info();

    let (width, height) = (info.width, info.height);
    let buffer_size = info.raw_bytes();
    let bit_depth = info.bit_depth;
    let color_type = info.color_type;

    let path_out = Path::new(filename_out);
    let file = File::create(path_out).unwrap();
    let w = &mut BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height); // Width is 2 pixels and height is 1.
    encoder.set_color(color_type);
    encoder.set_depth(bit_depth);

    let mut buffer = vec![0; buffer_size];
    png_reader.next_frame(&mut buffer).unwrap();


    match bit_depth {
        BitDepth::Eight => {
            let (array_image, array2hist) = image_pair_from_buffer(buffer, height as usize, width as usize, color_type.samples(), params.denoise);

            let result_bytes = equalize_full_image(array_image, array2hist, method, params);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&result_bytes.into_iter().collect::<Vec<u8>>()).unwrap();
        }
        BitDepth::Sixteen => {
            let (h, w, cc) = (height as usize, width as usize, color_type.samples());
            let mut buffer_u16 = vec![0; h * w * cc];
            let mut buffer_cursor = Cursor::new(buffer);
            buffer_cursor
                .read_u16_into::<BigEndian>(&mut buffer_u16)
                .unwrap();
            let (array_image, array2hist) = image_pair_from_buffer(buffer_u16, height as usize, width as usize, color_type.samples(), params.denoise);
            let result_bytes = equalize_full_image(array_image, array2hist, method, params);

            let buffer_u8_le = result_bytes.into_iter().flat_map(|v: u16| v.to_be_bytes()).collect::<Vec<u8>>();
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&buffer_u8_le).unwrap();

        }
        v => {panic!("Supposed to be 8 or 16 bit image, got {v:?}")}
    }
}

fn load_8bit_img_as_array(filename: &str) -> (usize, Array3<u8>) { //Level, data
    let image = ImageReader::open(filename).unwrap().decode().unwrap();
    let (h, w) = (image.height() as usize, image.width() as usize);
    let (bm, cc) = match image {
        DynamicImage::ImageLuma8(i) => {(i.into_vec(), 1)},
        DynamicImage::ImageRgb8(i) => {(i.into_vec(), 3)},
        v => {panic!("Supposed to be 8 bit image, got {v:?}")}
    };
    (256, Array3::from_shape_vec((h, w, cc), bm).unwrap())
}


pub fn transform_any_8bit_image(filename_in: &str, filename_out: &str, method: Method, params: &HEParams) {
    let image = ImageReader::open(filename_in).unwrap().decode().unwrap();
    let (h, w) = (image.height(), image.width());
    let cc = match image {
        DynamicImage::ImageLuma8(_) => 1usize,
        DynamicImage::ImageRgb8(_) => 3,
        v => {panic!("Supposed to be 8 bit RGB ro GRAY image, got {v:?}")}
    };
    let data = image.as_bytes();

    let (array_image, array2hist) = image_pair_from_buffer(data.to_vec(), h as usize, w as usize, cc, params.denoise);

    let result_bytes = equalize_full_image(array_image, array2hist, method, params);

    let img: DynamicImage = match cc {
        3 => DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        1 => DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, result_bytes.into_iter().collect::<Vec<u8>>()).unwrap()),
        _ => unimplemented!()
    };
    img.save(filename_out).unwrap();
}

fn equalize_full_image<I>(array: Rc<Array3<I>>, array2hist: Rc<Array3<I>>, method: Method, params: &HEParams) -> Array3<I>
where I: HSLable + PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded + std::convert::TryFrom<i32>, u32: From<I>, f32: From<I>, usize: From<I>, u64: From<I>, i32: From<I>
{
    let max_val = u32::from(I::max_value());
    let (h, w, cc) = if let [hh, ww, ccc] = array2hist.shape() {
        (*hh, *ww, *ccc)
    } else {
        panic!("Supposed to be 3D array")
    };
    // Brightness level as max value of RGB
    let brightness: Array2<I> = if cc > 1 {
        array2hist.map_axis(Axis(2), |v| unsafe{ *v.iter().max().unwrap_unchecked() })
    } else {
        array2hist.map_axis(Axis(2), |v|v[0])
    };

    let tuned_brightness: Array2<f32> = match method {
        Method::HE_HSL => {he_2d_hsl(array2hist.as_ref(), usize::from(I::max_value())+1, params)}
        Method::CLAHE_HSL => {clahe_2d_hsl(array2hist.as_ref(), params)}
        Method::HE => {he_2d::<I,JustHist>(&brightness, params)}
        Method::CLAHE => {clahe_2d::<I,JustHist>(&brightness, params)}
        Method::HE_NOISY => {he_2d::<I,NoisyHist>(&brightness, params)}
        Method::CLAHE_NOISY => {clahe_2d::<I,NoisyHist>(&brightness, params)}
    };
    let tuned_brightness = tuned_brightness.into_shape((h, w)).unwrap();
    let brightness_f: Array2<f32> = brightness.mapv(|v| f32::from(v.max(I::one()))).into_shape((h, w)).unwrap();

    //Save lima multiplier as image
    let mut brightness_relation_f: Array2<f32> = tuned_brightness.clone() / brightness_f.clone();
    let br_max = *brightness_relation_f.max().unwrap();

    let cf_brr = 255.0 / br_max;
    brightness_relation_f *= cf_brr;
    let brightness_relation = brightness_relation_f.into_iter().map(|v| v.round() as u8).collect::<Vec<u8>>();
    DynamicImage::ImageLuma8(GrayImage::from_raw(w as u32, h as u32, brightness_relation).unwrap())
        .save("/tmp/brightness_relation.png").unwrap();


    let tuned_brightness = tuned_brightness.into_shape((h, w, 1)).unwrap();
    let brightness_f = brightness_f.into_shape((h, w, 1)).unwrap();


    // Align full RGB image
    let result_float: Array3<f32> = array.mapv(|elem| f32::from(elem)) * tuned_brightness / brightness_f;
    let result_bytes: Array3<I> = result_float.mapv(|elem| I::from_u32((elem.round() as u32).min(max_val)).unwrap());

    result_bytes
}

fn he_2d<I,H>(img_array: &Array2<I>, params: &HEParams) -> Array2<f32>
where I: HSLable + PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded, usize: From<I>, f32: From<I>,
      H: Historator<I>
{
    let level: usize = I::max_value().as_();
    let mut hist = H::calc_hist(&img_array.view());
    let _max_hist = hist.max().unwrap().ceil() as usize;
    //dbg!(max_hist);
    //plot_histogram::plot(&format!("/tmp/plots/single_orig.png"), &hist, max_hist);
    let treschold_int = (level + 1) / 32; //8 for u8
    clip_hist(&mut hist, treschold_int as f32);
    let _max_hist_clipped = hist.max().unwrap().ceil() as usize;
    //dbg!(max_hist_clipped);
    //plot_histogram::plot(&format!("/tmp/plots/single_clip.png"), &hist, max_hist_clipped);
    let hist_cdf: Array1<f32> = calc_hist_cdf(&hist, usize::from(I::max_value())+1, params.limits());
    //plot_histogram::plot(&format!("/tmp/plots/single_cdf.png"), &hist_cdf, level);

    
    img_array.mapv(|v: I| hist_cdf[usize::from(v)])
}

fn clahe_2d<I,H>(img_array: &Array2<I>, params: &HEParams) -> Array2<f32>
where I: HSLable + PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded + std::convert::TryFrom<i32>, usize: From<I>, u64: From<I>, i32: From<I>, f32: From<I>,
      H: Historator<I>
{
    let (w, h) = if let [mm, nn] = img_array.shape() {
        (*mm, *nn)
    } else {
        panic!("Supposed to be 2D array")
    };

    let blocks_h = params.blocks().size_h;
    let blocks_w = params.blocks().size_w.unwrap_or_else(|| (blocks_h as f32 * w as f32 / h as f32).round() as usize);

    let block_w = 2 * (w as f32 / (2 * blocks_w) as f32).ceil() as usize;
    let block_h = 2 * (h as f32 / (2 * blocks_h) as f32).ceil() as usize;

    let mut maps: Vec<Vec<Array1<f32>>> = vec![vec![]; blocks_w];
    let level = usize::from(I::max_value())+1;
    for i in 0..blocks_w {
        for j in 0..blocks_h {
            //block border
            let (si, ei) = (i * block_w, (i + 1) * block_w);
            let (sj, ej) = (j * block_h, (j + 1) * block_h);

            let block_view = img_array.slice(s![si..ei.min(w), sj..ej.min(h)]);

            //Switch hist method here

            let mut hist = H::calc_hist(&block_view);
            //plot_histogram::plot(&format!("/tmp/plots/{}_{}_orig.png", i, j), &hist, level);
            //let mut hist = calc_hist_noise(&block_view);
            //plot_histogram::plot(&format!("/tmp/plots/{}_{}_clip.png", i, j), &hist, level);
            let hist_cdf = calc_hist_cdf(&hist, level, params.limits());
            let treschold = level / 32;
            clip_hist(&mut hist, treschold as f32);
            //plot_histogram::plot(&format!("/tmp/plots/{}_{}_cdf.png", i, j), &hist_cdf, level);
            maps[i].push(hist_cdf);
        }
    }


    let block_m = block_w as isize;
    let block_n = block_h as isize;

    let block_m_step = block_m / 2;
    let block_n_step = block_n / 2;

    let mut array_result: Array2<f32> = Array2::zeros((w, h,));

    let iblocks_h = blocks_h as isize;
    let iblocks_w = blocks_w as isize;

    for m_start in (0..w as isize).step_by(block_m_step as usize) {
        for n_start in (0..h as isize).step_by(block_n_step as usize) {
            let range_i = m_start..(m_start + block_m_step).min(w as isize);
            let range_j = n_start..(n_start + block_n_step).min(h as isize);
            let arr_i = Array1::from_iter(&mut range_i.clone());
            let arr_j = Array1::from_iter(&mut range_j.clone());

            let arr_r = (&arr_i - block_m_step) / block_m;
            let arr_c = (&arr_j - block_n_step) / block_n;

            let r = arr_r[0];
            let c = arr_c[0];

            let arr_x1: Array1<f32> = arr_i.mapv(|elem| elem as f32 / block_m as f32) - arr_r.mapv(|elem| elem as f32) - 0.5;
            let arr_y1: Array1<f32> = arr_j.mapv(|elem| elem as f32 / block_n as f32) - arr_c.mapv(|elem| elem as f32) - 0.5;

            let arr_x1_sub: Array1<f32> = 1.0 - &arr_x1;
            let arr_y1_sub: Array1<f32> = 1.0 - &arr_y1;

            let new_x_shape = (arr_x1.shape()[0], 1);

            let arr_x1 = arr_x1.into_shape(new_x_shape).unwrap();
            let arr_x1_sub = arr_x1_sub.into_shape(new_x_shape).unwrap();

            let img_tile = img_array.slice(s![range_i.clone(), range_j.clone()]);

            let corner_block = if r < 0 && c < 0 {
                Some(((r+1) as usize, (c+1) as usize))
            } else if r < 0 && c >=iblocks_w-1 {
                Some(((r+1) as usize, c as usize))
            } else if r > 0 && c < 0 {
                Some((r as usize, (c+1) as usize))
            } else if r >=iblocks_w-1 && c >=iblocks_h-1 {
                Some ((r as usize, c as usize))
            } else {
                None
            };

            if let Some((rl, cl)) = corner_block {
                //let img_tile = img_array.slice(s![range_i.clone(), range_j.clone()]);
                let mapped = img_tile.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                mapped.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if (r < iblocks_w-1 && c < iblocks_h-1) && (r >=0 && c >=0) {
                let rl = r as usize;
                let cl = c as usize;

                let mapped_lu = img_tile.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_lb = img_tile.mapv(|elem| maps[rl + 1][cl][elem.to_usize().unwrap()]);
                let mapped_ru = img_tile.mapv(|elem| maps[rl][cl + 1][elem.to_usize().unwrap()]);
                let mapped_rb = img_tile.mapv(|elem| maps[rl + 1][cl + 1][elem.to_usize().unwrap()]);

                let xs_mlu = &arr_x1_sub * mapped_lu;
                let x_mlb = &arr_x1 * mapped_lb;

                let xs_mru = &arr_x1_sub * mapped_ru;
                let x_mrb = &arr_x1 * mapped_rb;

                let mapped_mult_sum: Array2<f32> = arr_y1_sub * (xs_mlu + x_mlb) + arr_y1 * (xs_mru + x_mrb);
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if r < 0 || r >=iblocks_w-1 {
                let rl = r.max(0).min(iblocks_w-1) as usize;
                let cl = c as usize;
                let mapped_left = arr_y1_sub * img_tile.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_right = arr_y1 * img_tile.mapv(|elem| maps[rl][cl+1][elem.to_usize().unwrap()]);
                let mapped_mult_sum = mapped_left + mapped_right;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if c < 0 || c >=iblocks_h-1 {
                let rl = r as usize;
                let cl = c.max(0).min(iblocks_h-1) as usize;
                let mapped_up = arr_x1_sub * img_tile.mapv(|elem| maps[rl][cl][elem.to_usize().unwrap()]);
                let mapped_bottom = arr_x1 * img_tile.mapv(|elem| maps[rl+1][cl][elem.to_usize().unwrap()]);
                let mapped_mult_sum = mapped_up + mapped_bottom;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else {
                panic!("Should not be reached! r={r}, c={c}, iblocks_h={iblocks_h}, iblocks_w={iblocks_w}")
            }
        }
    }
    array_result
}


fn clip_hist(hist: &mut Array1<f32>, threshold: f32) {
    let all_sum = hist.sum();
    let threshold_value = threshold * all_sum / hist.len() as f32;
    let total_extra: f32 = hist.iter().filter(|v| v >= &&threshold_value).map(|v| v - threshold_value).sum();
    let mean_extra = total_extra / hist.len() as f32;

    hist.map_mut(|v: &mut f32| if *v >= threshold_value {*v = threshold_value + mean_extra} else {*v += mean_extra});
}

trait Historator<I> {
    fn calc_hist(img_array: &ArrayView2<I>) -> Array1<f32>;
}

struct JustHist {}
struct NoisyHist {}

impl<I: HSLable + std::convert::TryFrom<i32>> Historator<I> for JustHist {
    fn calc_hist(img_array: &ArrayView2<I>) -> Array1<f32> {
        calc_hist(img_array)
    }
}
impl<I: HSLable + Ord + std::convert::TryFrom<i32>> Historator<I> for NoisyHist {
    fn calc_hist(img_array: &ArrayView2<I>) -> Array1<f32> {
        calc_hist_noise(img_array)
    }
}

fn calc_hist<I>(img_array: &ArrayView2<I>) -> Array1<f32>
where I: HSLable
{
    let level: usize = I::max_value().as_();
    let level = level + 1;
    let mut hist: Array1<f32> = Array1::zeros((level,));
    img_array.for_each(|v| hist[v.to_usize().unwrap()] += 1.0);
    hist
}



fn calc_hist_noise<I>(img_array: &ArrayView2<I>) -> Array1<f32>
where I: HSLable + Ord + std::convert::TryFrom<i32>
{
    let level: usize = I::max_value().as_();
    let level = level + 1;
    let shape = img_array.shape();
    let h = shape[0];
    let w = shape[1];
    let total_px = (h * w) as f64;
    let mut hist: Array1<f32> = Array1::zeros((level,));
    let noise = calc_local_noise(img_array);
    let avg_noise = (noise.fold(0u64, |b, x| b + x.round() as u64) as f64 / total_px) as f32;
    let max_noise = *noise.max().unwrap();
    let min_noise = *noise.min().unwrap();
    let noise_range = max_noise - min_noise;
    dbg!(avg_noise);
    Zip::from(&noise)
        .and(img_array)
        .for_each(|n, v| {
            let v_non_zero_int = *v.max(&I::one());
            let v_non_zero: f32 = v_non_zero_int.as_();
            let idx: usize = v.as_();
            let noise_diff = 1.0 + (n - min_noise) / noise_range; //1.0 for zero-noise, 2.0 - for max noise
            hist[idx] += v_non_zero / noise_diff;
        });
    hist
}

fn calc_local_noise<I>(img_array: &ArrayView2<I>) -> Array2<f32>
where I: HSLable + std::convert::TryFrom<i32>
{
    let mut result: Array2<f32> = img_array.mapv(|_| 1.0f32);// Array2::zeros(img_array.shape());

    //img_array.slice(s![range_i.clone(), range_j.clone()]);

    let shape = img_array.shape();
    let h = shape[0] as isize;
    let w = shape[1] as isize;

    for i in [-1isize, 0, 1] {
        for j in [-1isize, 0, 1] {
            if i==0 && j==0 {
                continue
            }

            let range_h_orig = i.max(0)..(h+i).min(h);
            let range_w_orig = j.max(0)..(w+j).min(w);
            let range_h_movd = (-i).max(0)..(h-i).min(h);
            let range_w_movd = (-j).max(0)..(w-j).min(w);

            let slice_orig = s![range_h_orig, range_w_orig];
            let slice_movd = s![range_h_movd, range_w_movd];

            let shifted_orig = img_array.slice(&slice_orig);
            let shifted_movd = img_array.slice(&slice_movd);

            let mut shifted_result: Array2<f32> = result.slice(&slice_orig).mapv(|x| x);

            Zip::from(&mut shifted_result)
                .and(&shifted_orig)
                .and(&shifted_movd)
                .for_each(|w, &x, &y| {
                    let xi: i32 = x.as_();
                    let yi: i32 = y.as_();
                    *w += (xi - yi).abs() as f32;
                });

            shifted_result.view().assign_to(result.slice_mut(&slice_orig));
        }
    }
    result
}

fn calc_hist_cdf(hist: &Array1<f32>, level: usize, limits: &BrightnessLimits) -> Array1<f32> {
    let first_nz = hist.iter().enumerate().find(|(_i, v)| v>&&0.0).unwrap().0.max(1);
    let last_nz = hist.iter().enumerate().rev().find(|(_i, v)| v>&&0.0).unwrap().0.max(1);
    let total_nz = hist.iter().filter(|v| v>&&0.0).count();
    let mut hist_cumsum: Array1<f32> = hist.iter().cloned().collect(); //.take(last_nz+1)
    let _length = hist_cumsum.len();
    //let last_nz_tst = [0.0,1.0,2.0,3.0,4.0,5.0].into_iter().enumerate().rev().find(|(i, v)| v>&0.0).unwrap().0;
    //dbg!(length, first_nz, last_nz, last_nz_tst);

    hist_cumsum.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    //Limit contrast range to near original

    let min_level = first_nz as f32 * limits.dark_limit;
    let max_level = last_nz as f32 + (level - 1 - last_nz) as f32 * limits.bright_limit;


    let actual_min = hist[first_nz];
    let actual_max = *hist_cumsum.max().unwrap();
    let upper_bound = (level - 1) as f32;

    hist_cumsum -= actual_min;

    let cf = (max_level - min_level) / (actual_max - actual_min).max(1.0);

    hist_cumsum *= cf;
    hist_cumsum += min_level;

    #[cfg(debug_assertions)]{
        let hist_max = *hist_cumsum.max().unwrap();
        let hist_min = *hist_cumsum.min().unwrap();
        assert!(hist_min>0.0);
        assert!(hist_max<=upper_bound);
    }

    hist_cumsum = hist_cumsum.mapv(|v|v.clamp(0.0, upper_bound));

    hist_cumsum
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GrayImage};
    use ndarray::{Array2, Axis};
    use crate::{calc_local_noise, load_8bit_img_as_array, Method, transform_any_8bit_image};

    #[test]
    fn test_local_noise() {
        let (_l, img) = load_8bit_img_as_array("car.jpg");
        let br_img: Array2<u8> = img.mean_axis(Axis(2)).unwrap();
        let noise_img = calc_local_noise(&br_img.view());
        let shape_out = noise_img.raw_dim();
        let buf = noise_img.into_iter().map(|v| v.round().max(255.0) as u8).collect::<Vec<u8>>();
        let img_out = DynamicImage::ImageLuma8(GrayImage::from_raw(shape_out[1] as u32, shape_out[0] as u32, buf).unwrap());
        img_out.save("car_noise.png").unwrap();
    }
}