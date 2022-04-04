use std::fmt::Debug;
use std::ops::{Add, AddAssign, Sub};
use itertools::Itertools;
use ndarray::{array, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, Axis, OwnedRepr, RawData};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Bounded, FromPrimitive, NumCast, PrimInt, ToPrimitive, Unsigned};

pub(crate) trait HSL<I> {
    ///Hue value
    fn hue(&self) -> I;
    fn saturation(&self) -> I;
    fn lightness(&self) -> I;
}

impl<I> HSL<I> for Array1<I> where I: Copy {
    fn hue(&self) -> I {
        unsafe { *self.get(0).unwrap_unchecked() }
    }
    fn saturation(&self) -> I {
        unsafe { *self.get(1).unwrap_unchecked() }
    }
    fn lightness(&self) -> I {
        unsafe { *self.get(2).unwrap_unchecked() }
    }
}

impl<I> HSL<I> for ArrayView1<'_, I> where I: Copy {
    fn hue(&self) -> I {
        unsafe { *self.get(0).unwrap_unchecked() }
    }
    fn saturation(&self) -> I {
        unsafe { *self.get(1).unwrap_unchecked() }
    }
    fn lightness(&self) -> I {
        unsafe { *self.get(2).unwrap_unchecked() }
    }
}

impl<I> HSL<I> for ArrayView1<'_, &I> where I: Copy {
    fn hue(&self) -> I {
        unsafe { **self.get(0).unwrap_unchecked() }
    }
    fn saturation(&self) -> I {
        unsafe { **self.get(1).unwrap_unchecked() }
    }
    fn lightness(&self) -> I {
        unsafe { **self.get(2).unwrap_unchecked() }
    }
}

pub(crate) trait HSLable:
PartialEq + PartialOrd + ToPrimitive + Copy + Sub<Output = Self> + Add<Output = Self> + AddAssign
{

}

impl HSLable for u8 {}
impl HSLable for u16 {}
impl HSLable for i16 {}
impl HSLable for u32 {}
impl HSLable for i32 {}
impl HSLable for f32 {}

fn calc_single_row_hue<I>(row: ArrayView2<I>) -> Array2<f32>
where I: HSLable
{
    let values = [
        [1,2,0], //r is max, g - b, shift = 0
        [2,0,2], //g is max, b - r, shift = 2
        [0,1,4], //b is max, r - g, shift = 4
    ];

    let (w, cc) = if let [ww, ccc] = row.shape().clone() {
        (*ww, *ccc)
    } else {
        panic!("Supposed to be an 3D array")
    };

    let mut result = Array2::default((w, cc,));

    row.axis_iter(Axis(0)).enumerate().for_each(|(i, rgb)| unsafe {
        let max_idx = rgb.argmax().unwrap_unchecked();
        let max_val_i = rgb[max_idx];
        let max_val_f = rgb[max_idx].to_f32().unwrap_unchecked();
        let min_val = *rgb.min().unwrap_unchecked();
        //dbg!(rgb.shape());
        let res = if min_val==max_val_i {
            array![0.0, 0.0, max_val_f]
        } else {
            let lvs = values[max_idx];
            let (ca, cb, shift) = (rgb[lvs[0]], rgb[lvs[1]], lvs[2] as f32);
            let saturation = (min_val.to_f32().unwrap_unchecked()) / max_val_f;
            let diff = (max_val_i - min_val).to_f32().unwrap_unchecked();
            let h = 60.0 * ( shift + ((ca).to_f32().unwrap_unchecked() - (cb).to_f32().unwrap_unchecked()) / diff );

            array![(360.0 + h) % 360.0, saturation, max_val_f]
            //array![(360.0 + h) % 360.0, saturation, 192.0]
        };
        res.assign_to(result.index_axis_mut(Axis(0), i));

    });
    result
}

pub(crate) fn calc_hue<I>(array: &Array3<I>) -> Array3<f32>
where I: HSLable
{
    let (h, w, cc) = if let [hh, ww, ccc] = array.shape().clone() {
        (*hh, *ww, *ccc)
    } else {
        panic!("Supposed to be an 3D array")
    };

    let mut result = Array3::default(( h, w, cc, ));

    for i in 0..h  {
        let row: ArrayView2<I> = array.index_axis(Axis(0), i);
        calc_single_row_hue(row).assign_to(result.index_axis_mut(Axis(0), i));
    }
    result
}



pub trait HueDist<T> {
    fn calc_hue_start(&self) -> u32;
    fn calc_hue_distance(&self) -> f32;
}

impl<T> HueDist<T> for T
where T: std::ops::Rem<Output = T>, T: Copy + NumCast + AsPrimitive<u32> + FromPrimitive + ToPrimitive + std::ops::Div<Output = T>
{
    fn calc_hue_start(&self) -> u32 {
        unsafe { <u32 as NumCast>::from(self.clone() / T::from(120u32).unwrap_unchecked()).unwrap_unchecked() }
    }
    fn calc_hue_distance(&self) -> f32 {
        unsafe { <f32 as NumCast>::from(self.clone() % T::from(120u32).unwrap_unchecked()).unwrap_unchecked() / 120.0f32 }
    }
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, GrayImage};
    use ndarray::{Array2, Array3, Axis};
    use crate::{calc_local_noise, load_8bit_img_as_array, Method, transform_any_8bit_image};
    use crate::line_up_colors::calc_hue;
    use crate::line_up_colors::HSL;

    #[test]
    fn test_hue_convertor() {
        let (l, img) = load_8bit_img_as_array("huetest.png");
        let hue_unaligled = calc_hue(&img);
        let hue_aligled = hue_unaligled
            .lanes(Axis(2))
            .into_iter()
            .map(|v| (255.0 * (180.0 - v.hue() % 180.0) / 180.0).round() as u8)
            //.map(|v| v[2].round() as u8)
            .collect::<Vec<u8>>();
        let shape_out = hue_unaligled.shape();
        let img_out = DynamicImage::ImageLuma8(GrayImage::from_raw(shape_out[1] as u32, shape_out[0] as u32, hue_aligled).unwrap());
        img_out.save("huetest_hue.png");
    }
}