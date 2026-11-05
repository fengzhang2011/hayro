use crate::Renderer;
use hayro_interpret::CacheKey;
use hayro_interpret::Paint;
use hayro_interpret::encode::{EncodedShadingType, texture_dimensions};
use hayro_interpret::gradient::SvgGradientKind;
use hayro_interpret::pattern::Pattern;
use hayro_interpret::util::x_y_advances;
use kurbo::{Affine, BezPath, Rect};
use std::sync::Arc;
use vello_cpu::color::{AlphaColor, DynamicColor, Srgb};
use vello_cpu::peniko::{ColorStop, Gradient, ImageQuality, ImageSampler};
use vello_cpu::{Image, ImageSource, PaintType, PixelMetadata, Pixmap, peniko};

impl Renderer<'_> {
    #[must_use]
    pub(super) fn set_paint(
        &mut self,
        paint: &Paint<'_>,
        path_bbox: impl Fn() -> Rect,
        is_stroke: bool,
    ) -> Option<BezPath> {
        let __prof = crate::prof::Scope::start(&crate::prof::T_PAINT);

        let mut paint_transform = Affine::IDENTITY;
        let mut clip_path = None;

        let paint: PaintType = match paint.clone() {
            Paint::Color(c) => {
                let c = c.to_rgba().to_rgba8();
                AlphaColor::from_rgba8(c[0], c[1], c[2], c[3]).into()
            }
            Paint::Pattern(p) => {
                let path_transform = self.ctx.transform();
                // The path extent in PAGE device space: the transform's
                // linear part is the same for every render of the page
                // (a tiled viewer only varies the translation), so this
                // extent - and anything cached under it - is render-
                // invariant, unlike the render-local bbox.
                let [pt_a, pt_b, pt_c, pt_d, pt_e, pt_f] = path_transform.as_coeffs();
                let page_transform = Affine::new([pt_a, pt_b, pt_c, pt_d, 0.0, 0.0]);
                let mut full_page_bbox = page_transform.transform_rect_bbox(path_bbox());

                if is_stroke {
                    // Try to account for stroke in bbox.
                    let (a1, a2) = x_y_advances(path_transform);
                    let factor = a1.length().max(a2.length()) * self.ctx.stroke().width;
                    full_page_bbox = full_page_bbox.inflate(factor, factor);
                }

                // The same extent in this render's local coordinates, and
                // the tile-clipped variant for fallbacks.
                let full_bbox =
                    Affine::translate((pt_e, pt_f)).transform_rect_bbox(full_page_bbox);
                let bbox = full_bbox.intersect(Rect::new(
                    0.0,
                    0.0,
                    self.ctx.width() as f64,
                    self.ctx.height() as f64,
                ));

                match *p {
                    Pattern::Shading(s) => {
                        const NATIVE_GRADIENT_TOLERANCE: f32 = 0.01;

                        clip_path = s.shading.clip_path.clone();
                        // The encode (mesh evaluation per vertex) is pure
                        // per (shading, linear matrix), so cache it in the
                        // shared object cache instead of re-running it for
                        // every render of the same page.
                        let encoded_arc: Option<
                            Arc<hayro_interpret::encode::EncodedShadingPattern>,
                        > = self.global.object_cache.get_or_insert_with_budgeted(
                            s.cache_key(),
                            || {
                                let bytes = 4 << 20;
                                Some((Arc::new(s.encode()), bytes))
                            },
                        );
                        let encoded_fallback;
                        let encoded: &hayro_interpret::encode::EncodedShadingPattern =
                            match &encoded_arc {
                                Some(arc) => arc.as_ref(),
                                None => {
                                    encoded_fallback = s.encode();
                                    &encoded_fallback
                                }
                            };

                        static NATIVE_COUNT: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        static TEXTURE_COUNT: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        static SAMPLE_US: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        if let EncodedShadingType::RadialAxial(gradient) = &encoded.shading_type
                            && let Some(native) =
                                gradient.as_svg_gradient(&encoded, bbox, NATIVE_GRADIENT_TOLERANCE)
                        {
                            paint_transform = path_transform.inverse()
                                * Affine::translate((-0.5, -0.5))
                                * native.transform;

                            let stops = native
                                .stops
                                .iter()
                                .map(|stop| ColorStop {
                                    offset: stop.offset,
                                    color: DynamicColor::from_alpha_color(AlphaColor::<Srgb>::new(
                                        stop.color,
                                    )),
                                })
                                .collect::<Vec<_>>();

                            let gradient = match native.kind {
                                SvgGradientKind::Linear { start, end } => {
                                    Gradient::new_linear(start, end)
                                }
                                SvgGradientKind::Radial {
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                } => Gradient::new_two_point_radial(
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                ),
                            }
                            .with_extend(peniko::Extend::Pad)
                            .with_stops(stops.as_slice());

                            PaintType::Gradient(gradient)
                        } else {
                            TEXTURE_COUNT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let __sample_start = std::time::Instant::now();
                            let _ = &__sample_start;
                            // Sample over the path's full extent (not the
                            // render-clipped bbox) when that extent is
                            // reasonable: the texture - and its cache key -
                            // are then identical for every render of this
                            // path, and each render paints its own window
                            // of the same texture. The encoded shading is
                            // translation-invariant, so this is exact.
                            // Vastly larger paths keep the clipped,
                            // uncached behavior.
                            let sample_bbox =
                                if full_page_bbox.width() * full_page_bbox.height()
                                    < 4.0
                                        * f64::from(self.ctx.width())
                                        * f64::from(self.ctx.height())
                                {
                                    full_bbox
                                } else {
                                    bbox
                                };
                            let (width, height) = texture_dimensions(full_page_bbox, 1.0);
                            let cached: Option<Arc<(u32, u32, Affine, Pixmap)>> = self
                                .global
                                .object_cache
                                .get_or_insert_with_budgeted(
                                    (s.cache_key(), (width, height)).cache_key(),
                                    || {
                                        let mut image =
                                            Vec::with_capacity(width as usize * height as usize);
                                        let (width, height, transform) =
                                            encoded.sample_texture(sample_bbox, 1.0, |sample| {
                                                let pixel = AlphaColor::<Srgb>::new(sample)
                                                    .premultiply()
                                                    .to_rgba8();
                                                image.push(pixel);
                                            });
                                        let pixmap = Pixmap::from_parts(
                                            bytemuck::cast_vec(image),
                                            width as u16,
                                            height as u16,
                                            PixelMetadata {
                                                may_have_transparency: true,
                                                ..Default::default()
                                            },
                                        );
                                        let bytes = pixmap.data().len() as u64;
                                        Some((
                                            Arc::new((width, height, transform, pixmap)),
                                            bytes,
                                        ))
                                    },
                                );
                            SAMPLE_US.fetch_add(
                                __sample_start.elapsed().as_micros() as u64,
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            if std::env::var("HAYRO_BENCH_DEBUG").is_ok() {
                                eprintln!(
                                    "[paint] texture={} sample_us={} key={:x}",
                                    TEXTURE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                                    SAMPLE_US.load(std::sync::atomic::Ordering::Relaxed),
                                    s.cache_key()
                                );
                            }
                            let Some((width, height, transform, pixmap)) =
                                cached.map(|arc| (*arc).clone())
                            else {
                                return None;
                            };
                            paint_transform = path_transform.inverse() * transform;

                            let image = Image {
                                image: ImageSource::Pixmap(Arc::new(pixmap)),
                                sampler: ImageSampler {
                                    x_extend: peniko::Extend::Repeat,
                                    y_extend: peniko::Extend::Repeat,
                                    quality: ImageQuality::Medium,
                                    alpha: 1.0,
                                },
                            };

                            PaintType::Image(image)
                        }
                    }
                    Pattern::Tiling(t) => {
                        let (image, transform) = self.tiling_paint(&t, bbox, is_stroke);
                        paint_transform = path_transform.inverse() * transform;

                        PaintType::Image(image)
                    }
                }
            }
        };

        self.ctx.set_paint_transform(paint_transform);
        self.ctx.set_paint(paint);

        clip_path
    }
}
