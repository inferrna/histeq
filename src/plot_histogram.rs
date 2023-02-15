use ndarray::Array1;
use ndarray_stats::QuantileExt;

use plotters::prelude::{AreaSeries, BLUE, ChartBuilder, Color, IntoDrawingArea, RED, BitMapBackend, WHITE};

pub(crate) fn plot(filename: &str, data: &Array1<f32>, max_value: usize) {
    let width = 2048;
    let height = 2048;
    dbg!(data.max().unwrap());
    dbg!(data.last().unwrap());
    let root = BitMapBackend::new(&filename, (width, height)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .margin(0i32)
        .build_cartesian_2d(0..data.len(), 0.0..max_value as f32).unwrap();

    chart.configure_mesh()
        .draw()
        .unwrap();

    chart.draw_series(
        AreaSeries::new(
            (0..).zip(data.iter()).map(|(x, y)| (x, *y)),
            0.0,
            RED.mix(0.2),
        ).border_style(BLUE),
    ).unwrap();
    root.present().expect("Unable to write result to file");
}