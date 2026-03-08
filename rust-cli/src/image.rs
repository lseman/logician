use std::path::Path;

use image::DynamicImage;
use ratatui::{layout::Rect, Frame};
use ratatui_image::{picker::Picker, protocol::StatefulProtocol, Resize, StatefulImage};

pub struct ImageRenderer {
    picker: Picker,
    state: Option<StatefulProtocol>,
}

impl ImageRenderer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            picker: picker_or_halfblocks(),
            state: None,
        })
    }

    pub fn load_path(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if !Path::new(path).exists() {
            return Err(format!("Image not found: {path}").into());
        }

        let img = load_image(path)?;
        self.state = Some(self.picker.new_resize_protocol(img));
        Ok(())
    }

    pub fn clear(&mut self) {
        self.state = None;
    }

    pub fn render_in_frame(
        &mut self,
        frame: &mut Frame,
        area: Rect,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| "No image has been loaded yet".to_string())?;
        let image = StatefulImage::default().resize(Resize::Fit(None));
        frame.render_stateful_widget(image, area, state);
        Ok(())
    }
}

pub fn load_image(path: &str) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    Ok(image::open(path)?)
}

fn picker_or_halfblocks() -> Picker {
    match Picker::from_query_stdio() {
        Ok(picker) => picker,
        Err(_) => Picker::halfblocks(),
    }
}
