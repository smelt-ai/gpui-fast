// Modified by Longbridge for gpui-fast.
//! Caller-managed GPU tile-cached layers for expensive opaque or transparent
//! subtrees keyed by [`CachedLayerId`] and a monotonic revision number.

use crate::{
    Bounds, ContentMask, Pixels, Point, Rgba, ScaledPixels, Scene, Size, TileCoord, Window,
    WindowBackgroundAppearance,
    fast::layers::{
        COMPILED, background,
        paint::{
            TILE_SIZE, all_tiles, draw_into_frame, insert_frame_tile_quads, insert_overlay,
            translated_scene, translation, viewports,
        },
        policy::DROP_AFTER_FRAMES,
        scene::{LayerContent, LayerFrame, LayerKey},
        tiles::{dirty_tiles, tile_hashes},
    },
    point,
    scene::{PaintOperation, Primitive},
};
use collections::FxHashMap;
use std::{mem, rc::Rc};

const TRANSPARENT_BLACK: Rgba = Rgba {
    r: 0.,
    g: 0.,
    b: 0.,
    a: 0.,
};

/// Stable caller-chosen identifier for a cached layer in a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CachedLayerId(u64);

impl CachedLayerId {
    /// Creates a cached layer identifier from a raw `u64` key.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw `u64` key of this cached layer identifier.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for CachedLayerId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<CachedLayerId> for u64 {
    fn from(id: CachedLayerId) -> Self {
        id.0
    }
}

/// Rendering options for [`Window::paint_cached_layer`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct CachedLayerOptions {
    /// Explicit tile clear color. When `None`, automatically bakes a solid
    /// opaque backdrop underneath the layer if available, or clears to
    /// transparent black and composites with premultiplied alpha.
    pub background: Option<Rgba>,
    /// Logical offset applied when positioning the layer content relative to
    /// `bounds.origin`. Changing only `offset` (or `bounds.origin`) with the
    /// same `revision` translates cached tiles without re-running `paint`.
    pub offset: Point<Pixels>,
}

impl CachedLayerOptions {
    /// Sets an explicit tile clear color.
    pub fn background(mut self, background: impl Into<Rgba>) -> Self {
        self.background = Some(background.into());
        self
    }

    /// Sets the logical translation offset for the layer content.
    pub fn offset(mut self, offset: Point<Pixels>) -> Self {
        self.offset = offset;
        self
    }
}

#[derive(Default)]
pub(crate) struct CachedLayers {
    pub(crate) entries: FxHashMap<CachedLayerId, CachedLayer>,
    scale_factor: Option<f32>,
}

pub(crate) struct CachedLayer {
    pub(crate) key: LayerKey,
    pub(crate) generation: u64,
    pub(crate) last_composited_frame: u64,
    pub(crate) record: Option<CachedLayerRecord>,
}

pub(crate) struct CachedLayerRecord {
    pub(crate) content: LayerContent,
    pub(crate) generation: u64,
    pub(crate) revision: u64,
    pub(crate) bounds_size: Size<Pixels>,
    pub(crate) scale_factor: f32,
    pub(crate) background: Rgba,
    pub(crate) tile_hashes: FxHashMap<TileCoord, u64>,
    pub(crate) dirty_tiles: Vec<TileCoord>,
    pub(crate) overlay: Rc<[Primitive]>,
    pub(crate) tileable_opaque: bool,
    pub(crate) has_subpixel: bool,
    pub(crate) tiled: bool,
}

struct CaptureGuard<'a> {
    window: &'a mut Window,
    saved_scene: Scene,
    active: bool,
}

impl<'a> CaptureGuard<'a> {
    fn begin(window: &'a mut Window, clip_bounds: Bounds<Pixels>) -> Self {
        window.content_mask_stack.push(ContentMask {
            bounds: clip_bounds,
        });
        let saved_scene = mem::take(&mut window.next_frame.scene);
        window.fast_layers.painting_cached = true;
        Self {
            window,
            saved_scene,
            active: true,
        }
    }

    fn finish(mut self) -> Scene {
        self.active = false;
        self.restore()
    }

    fn restore(&mut self) -> Scene {
        self.window.fast_layers.painting_cached = false;
        self.window.content_mask_stack.pop();
        mem::replace(
            &mut self.window.next_frame.scene,
            mem::take(&mut self.saved_scene),
        )
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            self.restore();
        }
    }
}

/// Paints or composites the cached layer `id` over `bounds`. Returns `Some`
/// when `paint` ran (first paint, revision change, size/scale change, or
/// fallback direct paint), or `None` when cached content was reused or clipped
/// away.
pub(crate) fn paint_cached_layer<R>(
    window: &mut Window,
    id: CachedLayerId,
    bounds: Bounds<Pixels>,
    revision: u64,
    options: CachedLayerOptions,
    paint: impl FnOnce(&mut Window) -> R,
) -> Option<R> {
    if !COMPILED
        || !window.fast_layers.enabled
        || window.fast_layers.painting_cached
        || window.fast_layers.painting.is_some()
    {
        return Some(paint(window));
    }
    if bounds.is_empty() {
        return None;
    }

    let frame = window.fast_layers.frame;
    let scale_factor = window.scale_factor();
    if window.fast_layers.cached.scale_factor != Some(scale_factor) {
        window.fast_layers.cached.entries.clear();
        window.fast_layers.cached.scale_factor = Some(scale_factor);
    }

    let clipped = bounds.intersect(&window.content_mask().bounds);
    if clipped.is_empty() {
        return None;
    }

    let clipped_viewport = window.cover_bounds(clipped);
    let cached_bg = window
        .fast_layers
        .cached
        .entries
        .get(&id)
        .and_then(|layer| layer.record.as_ref())
        .filter(|record| {
            record.revision == revision
                && record.scale_factor == scale_factor
                && record.bounds_size == bounds.size
        })
        .map(|record| record.background);

    let target_translation = translation(window, bounds.origin + options.offset);

    if let Some(cached_bg) = cached_bg {
        let next_bg = match options.background {
            Some(bg) => bg,
            None if cached_bg.a < 1.0 => cached_bg,
            None => bake_under(window, clipped_viewport).unwrap_or(TRANSPARENT_BLACK),
        };
        if let Some(layer) = window.fast_layers.cached.entries.get_mut(&id)
            && let Some(record) = layer.record.as_mut()
        {
            layer.last_composited_frame = frame;
            if next_bg != cached_bg {
                layer.generation += 1;
                let now_tiled =
                    record.tileable_opaque && (next_bg.a >= 1.0 || !record.has_subpixel);
                record.background = next_bg;
                record.generation = layer.generation;
                record.tiled = now_tiled;
                record.dirty_tiles = if now_tiled {
                    all_tiles(&record.tile_hashes)
                } else {
                    Vec::new()
                };
            }
            // The same generation keeps its dirty tiles: the renderer may not
            // have rendered the frame that made it (no drawable in the
            // background), and the tile cache ignores them for a generation
            // it has seen. Scroll layers keep theirs the same way.
        }
        emit_cached_layer(window, id, clipped, target_translation);
        return None;
    }

    let background = options
        .background
        .unwrap_or_else(|| bake_under(window, clipped_viewport).unwrap_or(TRANSPARENT_BLACK));

    let guard = CaptureGuard::begin(window, bounds);
    let result = paint(guard.window);
    let captured_scene = guard.finish();

    let to_content = point(
        ScaledPixels(-target_translation.x.0),
        ScaledPixels(-target_translation.y.0),
    );
    let content = translated_scene(&captured_scene, to_content);
    let (content, overlay) = crate::fast::layers::overlay::lift(content);

    let (has_paths, has_subpixel, has_external) = classify_content(&content, &overlay);
    let tileable_opaque = !has_paths && !has_external;
    let tiled = tileable_opaque && (background.a >= 1.0 || !has_subpixel);

    let bounds_scaled = window.cover_bounds(bounds);
    let region = Bounds {
        origin: bounds_scaled.origin + to_content,
        size: bounds_scaled.size,
    };
    let hashes = tile_hashes(&content, TILE_SIZE, region);

    if !window.fast_layers.cached.entries.contains_key(&id) {
        let key = window.fast_layers.alloc_key();
        window.fast_layers.cached.entries.insert(
            id,
            CachedLayer {
                key,
                generation: 0,
                last_composited_frame: frame,
                record: None,
            },
        );
    }

    let Some(layer) = window.fast_layers.cached.entries.get_mut(&id) else {
        return Some(result);
    };
    layer.last_composited_frame = frame;

    let reused_generation = layer
        .record
        .as_ref()
        .filter(|old| {
            old.tiled && tiled && old.scale_factor == scale_factor && old.background == background
        })
        .map(|old| {
            let dirty = dirty_tiles(&old.tile_hashes, &hashes);
            (old.generation, dirty, old.dirty_tiles.clone())
        });

    let (generation, dirty_tiles) = match reused_generation {
        // The same content: the same generation, with the dirty tiles it was
        // made with (see the revision hit above).
        Some((generation, dirty, kept)) if dirty.is_empty() => (generation, kept),
        Some((_, dirty, _)) => {
            layer.generation += 1;
            (layer.generation, dirty)
        }
        None => {
            layer.generation += 1;
            let dirty = if tiled {
                all_tiles(&hashes)
            } else {
                Vec::new()
            };
            (layer.generation, dirty)
        }
    };

    layer.record = Some(CachedLayerRecord {
        content: LayerContent::from(content),
        generation,
        revision,
        bounds_size: bounds.size,
        scale_factor,
        background,
        tile_hashes: hashes,
        dirty_tiles,
        overlay,
        tileable_opaque,
        has_subpixel,
        tiled,
    });

    emit_cached_layer(window, id, clipped, target_translation);
    Some(result)
}

/// Removes the cached layer `id` from `window`, releasing its retained scene
/// and stopping future tile emissions so GPU caches expire cleanly.
pub(crate) fn remove_cached_layer(window: &mut Window, id: CachedLayerId) {
    window.fast_layers.cached.entries.remove(&id);
}

/// Drops cached layers not composited for [`DROP_AFTER_FRAMES`].
pub(crate) fn finish_frame(layers: &mut super::WindowLayers) {
    let frame = layers.frame;
    layers
        .cached
        .entries
        .retain(|_, layer| frame < layer.last_composited_frame + DROP_AFTER_FRAMES);
}

fn bake_under(window: &Window, viewport: Bounds<ScaledPixels>) -> Option<Rgba> {
    let window_opaque =
        window.platform_window.background_appearance() == WindowBackgroundAppearance::Opaque;
    background::bake(&window.next_frame.scene, viewport, window_opaque)
}

fn classify_content(content: &Scene, overlay: &[Primitive]) -> (bool, bool, bool) {
    let mut has_paths = false;
    let mut has_subpixel = false;
    let mut has_external = false;
    for operation in &content.paint_operations {
        if let PaintOperation::Primitive(primitive) = operation {
            match primitive {
                Primitive::Path(_) => has_paths = true,
                Primitive::SubpixelSprite(_) => has_subpixel = true,
                Primitive::Surface(_) | Primitive::NativeView(_) => has_external = true,
                _ => {}
            }
        }
    }
    for primitive in overlay {
        match primitive {
            Primitive::SubpixelSprite(_) => has_subpixel = true,
            Primitive::Surface(_) | Primitive::NativeView(_) => has_external = true,
            _ => {}
        }
    }
    (has_paths, has_subpixel, has_external)
}

fn emit_cached_layer(
    window: &mut Window,
    id: CachedLayerId,
    clipped: Bounds<Pixels>,
    translation: Point<ScaledPixels>,
) {
    let Some((key, record)) = window
        .fast_layers
        .cached
        .entries
        .get(&id)
        .and_then(|layer| layer.record.as_ref().map(|record| (layer.key, record)))
    else {
        return;
    };

    let viewports = viewports(window);
    window
        .content_mask_stack
        .push(ContentMask { bounds: clipped });
    if !record.tiled {
        let content = record.content.clone();
        let overlay = record.overlay.clone();
        draw_into_frame(window, content.operations(), translation);
        insert_overlay(
            &mut window.next_frame.scene,
            &overlay,
            translation,
            viewports,
        );
        window.content_mask_stack.pop();
        return;
    }
    let frame = LayerFrame {
        key,
        generation: record.generation,
        background: record.background,
        tile_size: TILE_SIZE,
        content: record.content.clone(),
        dirty_tiles: record.dirty_tiles.clone(),
    };
    insert_frame_tile_quads(
        &mut window.next_frame.scene,
        frame,
        viewports.snapped,
        translation,
    );
    let overlay = record.overlay.clone();
    insert_overlay(
        &mut window.next_frame.scene,
        &overlay,
        translation,
        viewports,
    );
    window.content_mask_stack.pop();
}
