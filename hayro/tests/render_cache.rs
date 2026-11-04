//! Regression tests for the cross-render caches: a document rendered
//! repeatedly (a tiled renderer renders each page many times) must decode
//! each image once, and images whose soft mask has different dimensions
//! must not panic the prepared-paint path.
//!
//! Fixtures are built in-memory by a minimal PDF writer below (the image
//! streams are Flate-compressed with a dev-dependency so they exercise
//! the filter/decode pipeline — uncompressed streams bypass it), keeping
//! this diff source-only.

use std::io::Write as _;

use flate2::write::ZlibEncoder;
use flate2::Compression;
use hayro::hayro_interpret::Cache;
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use hayro::{PixmapSettings, RenderCache, RenderSettings, render};

fn zlib(data: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// A minimal one-page PDF whose page draws one Flate-compressed image
/// XObject through `q <cm> /Im0 Do Q`, plus optional extra operators.
fn image_pdf(
    img_w: u32,
    img_h: u32,
    rgb: &[u8],
    smask: Option<(u32, u32, Vec<u8>)>,
    extra_ops: &str,
) -> Vec<u8> {
    let rgb = zlib(rgb);
    let smask = smask.map(|(w, h, d)| (w, h, zlib(&d)));

    // Object 1: catalog, 2: pages, 3: page, 4: contents, 5: image,
    // optional 6: smask.
    let ops = format!("q 200 0 0 100 0 0 cm /Im0 Do Q {extra_ops}\n");
    let mut image_dict = format!(
        "/Type /XObject /Subtype /Image /Width {img_w} /Height {img_h} \
         /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode"
    );
    if smask.is_some() {
        image_dict.push_str(" /SMask 6 0 R");
    }

    let mut bodies: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>".to_vec(),
        format!("<< /Length {} >>\nstream\n{ops}endstream", ops.len()).into_bytes(),
        Vec::new(),
    ];
    let mut image_stream =
        format!("<< {image_dict} /Length {} >>\nstream\n", rgb.len()).into_bytes();
    image_stream.extend_from_slice(&rgb);
    image_stream.extend_from_slice(b"\nendstream");
    bodies[4] = image_stream;

    if let Some((mw, mh, data)) = &smask {
        let mut mask_stream = format!(
            "<< /Type /XObject /Subtype /Image /Width {mw} /Height {mh} \
             /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length {} >>\nstream\n",
            data.len()
        )
        .into_bytes();
        mask_stream.extend_from_slice(data);
        mask_stream.extend_from_slice(b"\nendstream");
        bodies.push(mask_stream);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n");
    let mut offsets = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_at = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", bodies.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for off in &offsets {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
            bodies.len() + 1
        )
        .as_bytes(),
    );
    out
}

fn simple_image_pdf() -> Vec<u8> {
    // A 64x32 gradient plus a red rectangle, so the page has image and
    // vector content.
    let mut rgb = Vec::with_capacity(64 * 32 * 3);
    for y in 0..32u32 {
        for x in 0..64u32 {
            rgb.extend_from_slice(&[(x * 4) as u8, (y * 8) as u8, 128]);
        }
    }
    image_pdf(64, 32, &rgb, None, "1 0 0 rg 80 40 40 20 re f")
}

fn mismatched_alpha_pdf() -> Vec<u8> {
    let mut rgb = Vec::with_capacity(64 * 32 * 3);
    for y in 0..32u32 {
        for x in 0..64u32 {
            rgb.extend_from_slice(&[(x * 4) as u8, (y * 8) as u8, 128]);
        }
    }
    // A 16x8 soft mask (different dimensions), mostly opaque with a
    // transparent stripe.
    let mut mask = vec![255u8; 16 * 8];
    for row in mask.chunks_exact_mut(16) {
        row[0] = 0;
    }
    image_pdf(64, 32, &rgb, Some((16, 8, mask)), "")
}

fn non_white(pixmap: &hayro::vello_cpu::Pixmap) -> usize {
    pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .filter(|px| px[..3] != [255, 255, 255])
        .count()
}

#[test]
fn image_pages_are_decoded_once_and_reused() {
    let pdf = Pdf::new(simple_image_pdf()).unwrap();
    let page = pdf.pages().get(0).unwrap();
    let cache = RenderCache::new_shared(Cache::new());
    let interp = InterpreterSettings::default();
    let settings = RenderSettings::default();
    let pixmap_settings = PixmapSettings { x_scale: 2.0, y_scale: 2.0, ..Default::default() };

    let first = render(page, &cache, &interp, &settings, &pixmap_settings);
    let (_, misses) = cache.interpreter_cache().object_cache_handle().budgeted_counters();

    let second = render(page, &cache, &interp, &settings, &pixmap_settings);
    let (hits, misses_after) = cache.interpreter_cache().object_cache_handle().budgeted_counters();

    assert!(misses >= 1, "the image decode must be a miss first (saw {misses})");
    assert!(hits >= 1, "the second render must hit the decode cache");
    assert_eq!(misses_after, misses, "no new decode for the same page");

    // Caching memoizes: identical pixels.
    assert_eq!(first.data_as_u8_slice(), second.data_as_u8_slice());
    assert!(non_white(&second) > 500, "the page must actually have image content");
}

#[test]
fn mismatched_alpha_dimensions_render_without_panicking() {
    let pdf = Pdf::new(mismatched_alpha_pdf()).unwrap();
    let page = pdf.pages().get(0).unwrap();
    let cache = RenderCache::new_shared(Cache::new());
    let interp = InterpreterSettings::default();
    let settings = RenderSettings::default();
    let pixmap_settings = PixmapSettings { x_scale: 2.0, y_scale: 2.0, ..Default::default() };

    let first = render(page, &cache, &interp, &settings, &pixmap_settings);
    let second = render(page, &cache, &interp, &settings, &pixmap_settings);

    assert!(non_white(&second) > 500, "the composited image must be visible");
    assert_eq!(first.data_as_u8_slice(), second.data_as_u8_slice());
}
