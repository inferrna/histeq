use std::fmt::Debug;

use itertools::Itertools;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView3, Axis, s, Zip};
use crate::{BlocksCount, BrightnessLimits, calc_hist_cdf, clip_hist, HEParams};
use crate::line_up_colors::{calc_hue, HSL, HSLable, HueDist};
use num_traits::{AsPrimitive, Bounded, Float, FromPrimitive, PrimInt, sign::Unsigned, ToPrimitive, Zero};

fn calc_hist_hsl(img_array: &ArrayView3<f32>, level: usize, params: &HEParams) -> Vec<Array1<f32>>
{
    let hist_rg: Array1<f32> = Array1::zeros((level,));
    let hist_gb: Array1<f32> = Array1::zeros((level,));
    let hist_br: Array1<f32> = Array1::zeros((level,));
    let hist_li: Array1<f32> = Array1::zeros((level,));

    let mut all_hue_arrays = vec![hist_rg, hist_gb, hist_br, hist_li];


    //Calc hist
    img_array.lanes(Axis(2))
             .into_iter()
             .for_each( |h| {
                 let v = h.lightness().to_usize().unwrap();
                 let saturated_val = h.saturation();
                 let colored_val = 1.0 - saturated_val;
                 all_hue_arrays[3][v] += saturated_val;
                 let idx = h.hue().calc_hue_start() as usize;
                 let dist2next = h.hue().calc_hue_distance();
                 all_hue_arrays[idx][v] += colored_val * (1.0 - dist2next);
                 all_hue_arrays[(idx+1) % 3][v] += colored_val * dist2next;
             });
    all_hue_arrays.iter_mut().enumerate().map(|(i, hist)| {
        //clip_hist(&mut a, 12.0);
        let possibly_nan = hist.iter().enumerate().find_position(|(_, v)| v.is_nan());
        if let Some((j, _)) = possibly_nan {
            panic!("Found NaN at {i},{j}");
        }
        let possibly_inf = hist.iter().enumerate().find_position(|(_, v)| v.is_infinite());
        if let Some((j, _)) = possibly_inf {
            panic!("Found Inf at {i},{j}");
        }
        let possibly_not_zeros = hist.iter().find(|v| !v.is_zero());
        if possibly_not_zeros.is_none() {
            eprintln!("Found full zeros at {i}");
            hist.clone()
        } else {
            #[cfg(debug_assertions)]{
                println!("<{i} orig");
                for (i, h) in hist.iter().enumerate() {
                    let hi = h.round() as u32;
                    if hi > 0 { print!("{i}:{},", hi) }
                }
                println!(" {i} orig>");
            }
            let hist_cdf = calc_hist_cdf(hist, level, params.limits());
            #[cfg(debug_assertions)]{
                println!("<{i} cdf");
                for (i, h) in hist_cdf.iter().enumerate() {
                    let hi = h.round() as u32;
                    if hi > 0 { print!("{i}:{},", hi) }
                }
                println!(" {i} cdf>");
            }

            hist_cdf
        }
    }).collect()
}

#[inline]
fn apply_hist_hsl<I: Copy>(h: &ArrayView1<I>, hists_cdf: &[Array1<I>]) -> I
where I: HSLable + Debug + Float + std::ops::Mul<Output = I> + HueDist<I>, f32: AsPrimitive<I>
{
    let v = h.lightness().to_usize().unwrap();
    let saturated_val = h.saturation();
    let colored_val = I::one() - saturated_val;
    //let h_lightness = &hists_cdf[3][v];
    let v_val = hists_cdf[3][v] * saturated_val;
    let hue_idx: usize = ( h.hue().calc_hue_start() ).as_();
    let dist2next: I = h.hue().calc_hue_distance();
    let hue0_val = hists_cdf[hue_idx % 3][v] * (I::one() - dist2next) * colored_val;
    let hue1_val = hists_cdf[(hue_idx+1) % 3][v] * dist2next * colored_val;
    v_val + hue0_val + hue1_val
}

pub(crate) fn he_2d_hsl<I>(img_array: &Array3<I>, level: usize, params: &HEParams) -> Array2<f32>
where I: HSLable
{
    let shape = img_array.shape();
    let h = shape[0];
    let w = shape[1];

    let hsl_arr = calc_hue(img_array);
    let hists_cdf = calc_hist_hsl(&hsl_arr.view(), level, params);
/*
    let hists_cdf: Vec<Array1<f32>> = clipped_hists.into_iter()
        .enumerate().map(|(i, hist)| {
            #[cfg(debug_assertions)]
            dbg!(i);
            calc_hist_cdf(&hist, level, params.limits())
        }).collect();
*/
    let result: Array2<f32> = Array2::from_shape_vec((h, w), hsl_arr
        .lanes(Axis(2))
        .into_iter()
        .map(|h| {
            apply_hist_hsl(&h, &hists_cdf)
        }).collect::<Vec<f32>>()).unwrap();
    result
}

pub fn clahe_2d_hsl<I>(img_array: &Array3<I>, params: &HEParams) -> Array2<f32>
//where I: HSLable
where I: HSLable + PrimInt + Unsigned + FromPrimitive + ToPrimitive + std::ops::AddAssign + Debug + Bounded + std::convert::TryFrom<i32>, usize: From<I>, u64: From<I>, i32: From<I>, f32: From<I>
{
    let (w, h) = if let [mm, nn, _cc] = img_array.shape() {
        (*mm, *nn)
    } else {
        panic!("Supposed to be 3D array")
    };
    let level = usize::from(I::max_value()) + usize::from(I::from(1u32).unwrap());

    let blocks_h = params.blocks().size_h;
    let blocks_w = params.blocks().size_w.unwrap_or_else(|| (blocks_h as f32 * w as f32 / h as f32).round() as usize);

    let block_m = 2 * (w as f32 / (2 * blocks_w) as f32).ceil() as usize;
    let block_n = 2 * (h as f32 / (2 * blocks_h) as f32).ceil() as usize;

    let mut maps: Vec<Vec<Vec<Array1<f32>>>> = vec![vec![vec![]; blocks_h]; blocks_w];
    let hsl_arr = calc_hue(img_array);

    for i in 0..blocks_w {
        for j in 0..blocks_h {
            //block border
            let (si, ei) = (i * block_m, (i + 1) * block_m);
            let (sj, ej) = (j * block_n, (j + 1) * block_n);

            let block_view_hsl = hsl_arr.slice(s![si..ei.min(w), sj..ej.min(h), 0..3usize]);

            //Switch hist method here

            let hist = calc_hist_hsl(&block_view_hsl, level, params);
            maps[i][j] = hist;
        }
    }


    let block_m = block_m as isize;
    let block_n = block_n as isize;

    let block_m_step = block_m / 2;
    let block_n_step = block_n / 2;

    let mut array_result: Array2<f32> = Array2::zeros((w, h, ));

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

            let hsl_tile = hsl_arr.slice(s![range_i.clone(), range_j.clone(), 0..3usize]);

            let corner_block = if r < 0 && c < 0 {
                Some(((r + 1) as usize, (c + 1) as usize))
            } else if r < 0 && c >= iblocks_h - 1 {
                Some(((r + 1) as usize, c as usize))
            } else if r > 0 && c < 0 {
                Some((r as usize, (c + 1) as usize))
            } else if r >= iblocks_w - 1 && c >= iblocks_h - 1 {
                Some((r as usize, c as usize))
            } else {
                None
            };

            if let Some((rl, cl)) = corner_block {
                Zip::from(hsl_tile.lanes(Axis(2)))
                        .and(array_result.slice_mut(s![range_i.clone(), range_j.clone()]))
                        .for_each(|h, v| *v = apply_hist_hsl(&h, &maps[rl][cl]));
                //mapped.assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone(), 0..3]).mapv_into());
            } else if (r < iblocks_w - 1 && c < iblocks_h - 1) && (r >= 0 && c >= 0) {
                let rl = r as usize;
                let cl = c as usize;

                let mapped_lu = Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl][cl]) );
                let mapped_lb = Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl + 1][cl]) );
                let mapped_ru = Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl][cl + 1]) );
                let mapped_rb = Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl + 1][cl + 1]) );

                let xs_mlu = &arr_x1_sub * mapped_lu;
                let x_mlb = &arr_x1 * mapped_lb;

                let xs_mru = &arr_x1_sub * mapped_ru;
                let x_mrb = &arr_x1 * mapped_rb;

                let mapped_mult_sum: Array2<f32> = arr_y1_sub * (xs_mlu + x_mlb) + arr_y1 * (xs_mru + x_mrb);
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if r < 0 || r >= iblocks_w - 1 {
                let rl = r.max(0).min(iblocks_w - 1) as usize;
                let cl = c as usize;
                let mapped_left = arr_y1_sub * Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl][cl]) );
                let mapped_right = arr_y1 * Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl][cl +1 ]) );
                let mapped_mult_sum = mapped_left + mapped_right;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else if c < 0 || c >= iblocks_h - 1 {
                let rl = r as usize;
                let cl = c.max(0).min(iblocks_h - 1) as usize;
                let mapped_up = arr_x1_sub * Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl][cl]) );
                let mapped_bottom = arr_x1 * Zip::from(hsl_tile.lanes(Axis(2))).map_collect(|h| apply_hist_hsl(&h, &maps[rl + 1][cl]) );
                let mapped_mult_sum = mapped_up + mapped_bottom;
                mapped_mult_sum.view().assign_to(array_result.slice_mut(s![range_i.clone(), range_j.clone()]));
            } else {
                panic!("Should not be reached! r={r}, c={c}, iblocks_h={iblocks_h}, iblocks_w={iblocks_w}")
            }
        }
    }
    array_result
}