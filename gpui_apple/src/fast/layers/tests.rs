// Modified by Longbridge for gpui-fast.
//! Pixel tests of scroll layers on Metal: the same content drawn directly and
//! through layer tiles must come out byte for byte the same. They follow
//! `gpui_wgpu`'s `fast::layers::tests`.
//!
//! [`Harness`] draws scenes with a headless `MetalRenderer`, through
//! `render_frame` as a window's frames are drawn, into a texture it reads
//! back. Without a Metal device the tests print a skip and return.

use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;

use gpui::{
    AtlasKey, AtlasTile, Bounds, ContentMask, Corners, DevicePixels, Edges, FontId, GlyphId, Hsla,
    ImageId, LayerFrame, LayerKey, MonochromeSprite, PaintedPath, PathBuilder, PlatformAtlas,
    PolychromeSprite, Quad, RenderGlyphParams, RenderImageParams, Rgba, ScaledPixels, Scene,
    SceneLayers, Shadow, Size, TileCoord, TransformationMatrix, Underline, decode_layer_tile,
    layer_tile_id, layer_tile_texture_id, linear_color_stop, linear_gradient, point, px, rgba,
    size,
};
use objc2_metal::{MTLCommandBuffer as _, MTLDevice as _};
use parking_lot::Mutex;

use crate::fast::layers::TileCache;
use crate::metal_renderer::{InstanceBufferPool, MetalRenderer};

const TILE: u32 = 512;

#[path = "../../../../gpui/src/fast/rounded_clip_fixtures.rs"]
mod rounded_clip_tests;

#[gpui::test]
fn rounded_overflow_retained_child_follows_ancestor_clip(cx: &mut gpui::TestAppContext) {
    let mut harness = Harness::new().expect("retained clipping regression requires a GPU");
    rounded_clip_tests::retained_frames(cx, |scene, size| harness.render(scene, size));
}

fn render_div_fixture(cx: &mut gpui::TestAppContext, build: fn() -> gpui::Div) -> (Vec<u8>, usize) {
    let mut harness = Harness::new().expect("rounded clipping regression requires a Metal device");
    rounded_clip_tests::render_with(cx, build, |scene, size| harness.render(scene, size))
}

#[gpui::test]
fn rounded_overflow_inside_scroll_tiles_matches_direct_pixels(cx: &mut gpui::TestAppContext) {
    let mut harness = Harness::new().expect("rounded clipping regression requires a Metal device");
    rounded_clip_tests::render_with(cx, rounded_clip_tests::square_fixture, |scene, size| {
        let direct = harness.render(scene, size);
        let tiled = rounded_clip_tests::tiled_scene(scene, size);
        let actual = harness.render(&tiled, size);
        assert_same_pixels(&actual, &direct, size.width.0 as usize);
        actual
    });
}

/// Manual timing reuses a private target and waits for GPU completion without
/// reading pixels back. Compare the
/// same 64 descendants with and without curved overflow, at two target sizes,
/// to expose full-viewport mask work even when the clipped elements are small.
#[gpui::test]
#[ignore = "manual GPU performance measurement"]
fn rounded_overflow_mask_cost(cx: &mut gpui::TestAppContext) {
    fn rectangular() -> gpui::Div {
        use gpui::{ParentElement as _, Styled as _};
        gpui::div()
            .flex()
            .flex_wrap()
            .size(px(64.))
            .bg(gpui::white())
            .children((0..64).map(|_| {
                gpui::div()
                    .size(px(8.))
                    .overflow_hidden()
                    .child(gpui::div().size(px(8.)).bg(gpui::black()))
            }))
    }
    for side in [128, 1024] {
        for (name, build) in [
            ("rectangle", rectangular as fn() -> gpui::Div),
            ("rounded", rounded_clip_tests::sibling_fixture),
        ] {
            let mut harness = Harness::new().expect("timing requires Metal");
            rounded_clip_tests::render_with(cx, build, |scene, _| {
                let extent = device_size(side, side);
                let descriptor = objc2_metal::MTLTextureDescriptor::new();
                // SAFETY: The side is positive and this descriptor creates a 2D texture.
                unsafe {
                    descriptor.setWidth(side as usize);
                    descriptor.setHeight(side as usize);
                }
                descriptor.setPixelFormat(objc2_metal::MTLPixelFormat::BGRA8Unorm);
                descriptor.setStorageMode(objc2_metal::MTLStorageMode::Private);
                descriptor.setUsage(objc2_metal::MTLTextureUsage::RenderTarget);
                let target = harness
                    .renderer
                    .device
                    .newTextureWithDescriptor(&descriptor)
                    .expect("timing target texture");
                let mut samples = Vec::new();
                for frame in 0..18 {
                    objc2::rc::autoreleasepool(|_| {
                        let start = std::time::Instant::now();
                        let commands = harness
                            .renderer
                            .render_frame(scene, &target, extent)
                            .unwrap();
                        let encoded = start.elapsed();
                        commands.commit();
                        commands.waitUntilCompleted();
                        if frame >= 3 {
                            samples.push((start.elapsed(), encoded));
                        }
                    });
                }
                samples.sort();
                eprintln!(
                    "64 {name} clips, {side}px target: median {:?}, encode {:?}",
                    samples[7].0, samples[7].1
                );
                Vec::new()
            });
        }
    }
}

#[test]
fn harness_renders_a_quad_at_the_right_pixels() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let mut scene = Scene::default();
    scene.insert_primitive(quad(sp(0., 0., 32., 32.), Hsla::from(rgba(0xffffffff))));
    scene.insert_primitive(quad(sp(4., 4., 8., 8.), Hsla::from(rgba(0xff0000ff))));
    scene.finish();
    let pixels = harness.render(&scene, device_size(32, 32));
    assert_eq!(pixel(&pixels, 32, 8, 8), [255, 0, 0, 255]);
    assert_eq!(pixel(&pixels, 32, 1, 1), [255, 255, 255, 255]);
}

/// Tiles composited where they were painted hold the content's own pixels,
/// across every tile edge.
#[test]
fn tiles_composited_in_place_equal_the_content_drawn_directly() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = Window {
        size: device_size(1024, 1024),
        viewport: sp(0., 0., 1024., 1024.),
    };
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    composite_and_compare(&mut harness, &window, (0., 0.), (0., 0.), &tiles, false);
}

#[test]
fn tiles_of_negative_coordinates_composite_like_positive_ones() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = Window {
        size: device_size(1024, 1024),
        viewport: sp(0., 0., 1024., 1024.),
    };
    let tiles = [coord(-1, -1), coord(0, -1), coord(-1, 0), coord(0, 0)];
    let side = TILE as f32;
    composite_and_compare(
        &mut harness,
        &window,
        (-side, -side),
        (side, side),
        &tiles,
        false,
    );
}

#[test]
fn a_composited_layer_equals_direct_drawing() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    composite_and_compare(
        &mut harness,
        &scrolled_window(),
        (0., 0.),
        (37., -91.),
        &tiles,
        false,
    );
}

/// A path crossing the tile boundary is binned into both tile render passes.
#[test]
fn a_path_inside_two_tiles_matches_direct_drawing() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = Window {
        size: device_size(1024, 512),
        viewport: sp(0., 0., 1024., 512.),
    };
    let tiles = [coord(0, 0), coord(1, 0)];
    let mut direct = Scene::default();
    direct.insert_primitive(panel(&window));
    add_path(&mut direct);
    direct.insert_primitive(scrollbar());
    direct.finish();
    let expected = harness.render(&direct, window.size);

    let mut content = Scene::default();
    add_path(&mut content);
    content.finish();
    let layer = layer_frame(LayerKey(5), 1, content, &tiles);
    let scene = composited(&window, layer, &tiles, (0., 0.), false);
    let actual = harness.render(&scene, window.size);
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        // Translating curves into tile space can round edge coverage by one level.
        assert!(
            actual.abs_diff(expected) <= 1,
            "byte {index}: {actual} vs {expected}"
        );
    }
}

/// A path drawn directly over the composited layer.
#[test]
fn a_path_over_a_composited_layer_equals_direct_drawing() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    composite_and_compare(
        &mut harness,
        &scrolled_window(),
        (0., 0.),
        (37., -91.),
        &tiles,
        true,
    );
}

/// Paths something draws over stay out of the tiles: the core draws them,
/// and what it draws over them, over the composited tiles
/// (`gpui::fast::layers::overlay`), as without a layer. That holds at odd
/// translations too, as the paths are rasterized where they show.
#[test]
fn covered_paths_drawn_over_the_tiles_equal_direct_drawing() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = scrolled_window();
    let width = window.size.width.0 as usize;
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    for (generation, translation) in [(37., -91.), (38., -92.), (3., 5.), (101., -233.)]
        .into_iter()
        .enumerate()
    {
        let mut direct = Scene::default();
        direct.insert_primitive(panel(&window));
        add_content(&mut direct, &harness, translation, Some(window.viewport));
        add_covered_paths(&mut direct, &harness, translation, window.viewport);
        direct.insert_primitive(scrollbar());
        direct.finish();
        let expected = harness.render(&direct, window.size);

        let layer = layer_frame(
            LayerKey(7),
            generation as u64 + 1,
            content(&harness, (0., 0.), None),
            &tiles,
        );
        let mut scene = Scene::default();
        scene.insert_primitive(panel(&window));
        scene.push_layer(window.viewport);
        for &tile in &tiles {
            scene.insert_primitive(tile_sprite(
                layer.key,
                &layer,
                tile,
                translation,
                window.viewport,
            ));
        }
        scene.pop_layer();
        add_covered_paths(&mut scene, &harness, translation, window.viewport);
        scene.insert_primitive(scrollbar());
        scene.layers.frames.push(layer);
        scene.finish();
        let actual = harness.render(&scene, window.size);
        assert_same_pixels(&actual, &expected, width);
        // A frame that only scrolled draws the cached tiles again.
        let actual = harness.render(&scene, window.size);
        assert_same_pixels(&actual, &expected, width);
    }
}

/// A repainted layer's next generation re-rasterizes its dirty tile, which
/// then shows the new content.
#[test]
fn a_repainted_tile_shows_the_new_content() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = scrolled_window();
    let translation = (37., -91.);
    let key = LayerKey(6);
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];

    let first = layer_frame(key, 1, content(&harness, (0., 0.), None), &tiles);
    harness.render(
        &composited(&window, first, &tiles, translation, false),
        window.size,
    );
    let rasterized = harness.cache().rasterized;

    // The content with a green square in tile (0, 0) only.
    fn changed(
        scene: &mut Scene,
        harness: &Harness,
        offset: (f32, f32),
        clip: Option<Bounds<ScaledPixels>>,
    ) {
        add_content(scene, harness, offset, clip);
        let mut square = quad(
            sp(200. + offset.0, 300. + offset.1, 40., 40.),
            Hsla::from(rgba(0x00ff00ff)),
        );
        if let Some(clip) = clip {
            square.content_mask = ContentMask { bounds: clip };
        }
        scene.insert_primitive(square);
    }
    let mut new_content = Scene::default();
    changed(&mut new_content, &harness, (0., 0.), None);
    new_content.finish();
    let second = layer_frame(key, 2, new_content, &[coord(0, 0)]);
    let actual = harness.render(
        &composited(&window, second, &tiles, translation, false),
        window.size,
    );
    assert_eq!(harness.cache().rasterized, rasterized + 1);

    let mut direct = Scene::default();
    direct.insert_primitive(panel(&window));
    changed(&mut direct, &harness, translation, Some(window.viewport));
    direct.insert_primitive(scrollbar());
    direct.finish();
    let expected = harness.render(&direct, window.size);
    assert_same_pixels(&actual, &expected, window.size.width.0 as usize);
}

/// A tile sprite of a layer the cache holds nothing of draws nothing, and
/// never reaches the atlas.
#[test]
fn a_tile_the_cache_lacks_draws_nothing() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = scrolled_window();
    let mut expected = Scene::default();
    expected.insert_primitive(panel(&window));
    expected.insert_primitive(scrollbar());
    expected.finish();
    let expected = harness.render(&expected, window.size);

    // The scene composites tiles of layer 9 but describes layer 8 only.
    let layer = layer_frame(LayerKey(8), 1, Scene::default(), &[]);
    let mut scene = Scene::default();
    scene.insert_primitive(panel(&window));
    scene.push_layer(window.viewport);
    for tile in [coord(0, 0), coord(1, 0)] {
        scene.insert_primitive(tile_sprite(
            LayerKey(9),
            &layer,
            tile,
            (37., -91.),
            window.viewport,
        ));
    }
    scene.pop_layer();
    scene.insert_primitive(scrollbar());
    scene.layers.frames.push(layer);
    scene.finish();
    let actual = harness.render(&scene, window.size);
    assert_same_pixels(&actual, &expected, window.size.width.0 as usize);
}

#[test]
fn resize_releases_every_tile() {
    let Some(mut harness) = Harness::new() else {
        return;
    };
    let window = scrolled_window();
    let tiles = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    composite_and_compare(&mut harness, &window, (0., 0.), (37., -91.), &tiles, false);
    assert!(harness.cache().holds(LayerKey(5), coord(0, 0)));

    harness.renderer.update_drawable_size(window.size);
    assert!(harness.cache().is_empty());

    // The next frame rasterizes the tiles it composites again.
    let rasterized = harness.cache().rasterized;
    composite_and_compare(&mut harness, &window, (0., 0.), (37., -91.), &tiles, false);
    assert!(harness.cache().rasterized > rasterized);
}

#[test]
fn dirty_tiles_are_rasterized_once_they_are_shown() {
    let mut cache = TileCache::default();
    let key = LayerKey(3);
    let all = [coord(0, 0), coord(1, 0), coord(0, 1), coord(1, 1)];
    let mut rasterized = 0;
    let mut frame =
        |cache: &mut TileCache, generation, dirty: &[TileCoord], shown: &[TileCoord]| {
            let layers = scene_layers(key, generation, dirty);
            let planned = cache.begin_frame(&layers, shown.iter().map(|tile| (key, *tile)));
            rasterized += planned.len();
            planned
                .into_iter()
                .map(|(_, tile)| tile)
                .collect::<Vec<_>>()
        };

    let [a, b, c, _] = all;
    // A new layer: only the dirty tiles shown. The others wait until they are.
    let mut shown = vec![a, b];
    shown.sort();
    assert_eq!(frame(&mut cache, 1, &all, &[a, b]), shown);
    // The same generation again: nothing.
    assert_eq!(frame(&mut cache, 1, &all, &[a, b]), vec![]);
    // The next generation dirties a held tile that is not shown: it waits.
    assert_eq!(frame(&mut cache, 2, &[b, c], &[a]), vec![]);
    // Shown again, it is rasterized, and only it.
    assert_eq!(frame(&mut cache, 2, &[b, c], &[a, b]), vec![b]);
    // A dirty tile never shown before is rasterized once it is.
    assert_eq!(frame(&mut cache, 3, &[], &[a, b, c]), vec![c]);
    // A skipped generation may have dirtied any tile: shown tiles again.
    assert_eq!(frame(&mut cache, 5, &[], &[a]), vec![a]);
    // A shown tile the cache never had.
    assert_eq!(frame(&mut cache, 5, &[], &[coord(2, 0)]), vec![coord(2, 0)]);
    assert_eq!(rasterized, 6);
}

#[test]
fn tiles_are_evicted_least_recently_composited_first_within_the_budget() {
    const SIDE: u32 = 16;
    let tile_bytes = (SIDE * SIDE * 4) as u64;
    let key = LayerKey(4);
    let mut cache = TileCache::default();
    let layers = SceneLayers {
        frames: vec![LayerFrame {
            tile_size: SIDE,
            ..layer_frame(key, 1, Scene::default(), &[])
        }],
    };
    let frame = |cache: &mut TileCache, shown: &[TileCoord], budget: u64| {
        cache.begin_frame(&layers, shown.iter().map(|tile| (key, *tile)));
        cache.evict_to_budget(budget);
    };
    let (a, b, c, d, e) = (
        coord(0, 0),
        coord(1, 0),
        coord(2, 0),
        coord(3, 0),
        coord(4, 0),
    );

    frame(&mut cache, &[a, b, c], 3 * tile_bytes);
    frame(&mut cache, &[b, c, d], 3 * tile_bytes);
    assert!(
        !cache.holds(key, a),
        "the least recently composited tile goes"
    );
    assert!(cache.holds(key, b) && cache.holds(key, c) && cache.holds(key, d));

    frame(&mut cache, &[c], 3 * tile_bytes);
    frame(&mut cache, &[d, e], tile_bytes);
    assert!(!cache.holds(key, b) && !cache.holds(key, c));
    assert!(
        cache.holds(key, d) && cache.holds(key, e),
        "tiles composited this frame stay, over budget or not"
    );
}

#[test]
fn tile_sprites_split_into_runs_of_one_tile() {
    let mut cache = TileCache::default();
    let key = LayerKey(2);
    let layer = layer_frame(key, 1, Scene::default(), &[]);
    let viewport = sp(0., 0., 1024., 1024.);
    let mut scene = Scene::default();
    for tile in [coord(0, 0), coord(1, 0), coord(0, 1)] {
        scene.insert_primitive(tile_sprite(key, &layer, tile, (0., 0.), viewport));
    }
    scene.layers.frames.push(layer);
    scene.finish();
    cache.note_composited(&scene);
    let runs: Vec<(TileCoord, std::ops::Range<usize>)> = cache.tile_runs(0..3).collect();
    assert_eq!(runs.len(), 3);
    for (tile, run) in &runs {
        assert_eq!(run.len(), 1);
        let sprite = &scene.polychrome_sprites[run.start];
        assert_eq!(
            decode_layer_tile(sprite.tile.texture_id, sprite.tile.tile_id),
            Some((key, *tile))
        );
    }
    assert_eq!(cache.tile_runs(1..2).count(), 1);
}

// --- layer helpers --- //

/// A window of `size` whose scroll container shows its layer in `viewport`.
struct Window {
    size: Size<DevicePixels>,
    viewport: Bounds<ScaledPixels>,
}

fn scrolled_window() -> Window {
    Window {
        size: device_size(700, 500),
        viewport: sp(40., 30., 600., 400.),
    }
}

fn panel(window: &Window) -> Quad {
    quad(
        sp(
            0.,
            0.,
            window.size.width.0 as f32,
            window.size.height.0 as f32,
        ),
        Hsla::from(background()),
    )
}

fn scrollbar() -> Quad {
    quad(sp(620., 40., 12., 120.), Hsla::from(rgba(0x888888ff)))
}

fn coord(x: i32, y: i32) -> TileCoord {
    TileCoord { x, y }
}

fn background() -> Rgba {
    rgba(0x336699ff)
}

fn layer_frame(key: LayerKey, generation: u64, content: Scene, dirty: &[TileCoord]) -> LayerFrame {
    LayerFrame {
        key,
        generation,
        background: background(),
        tile_size: TILE,
        content: Rc::new(content).into(),
        dirty_tiles: dirty.to_vec(),
    }
}

fn scene_layers(key: LayerKey, generation: u64, dirty: &[TileCoord]) -> SceneLayers {
    SceneLayers {
        frames: vec![layer_frame(key, generation, Scene::default(), dirty)],
    }
}

/// The sprite that composites `tile` of `layer`'s content, under the key
/// `key`, moved by `translation` and clipped to `viewport`.
fn tile_sprite(
    key: LayerKey,
    layer: &LayerFrame,
    tile: TileCoord,
    translation: (f32, f32),
    viewport: Bounds<ScaledPixels>,
) -> PolychromeSprite {
    let bounds = layer.tile_bounds(tile);
    PolychromeSprite {
        order: 0,
        pad: 0,
        grayscale: false.into(),
        opacity: 1.,
        bounds: sp(
            bounds.origin.x.0 + translation.0,
            bounds.origin.y.0 + translation.1,
            TILE as f32,
            TILE as f32,
        ),
        content_mask: ContentMask { bounds: viewport },
        corner_radii: Corners::default(),
        tile: AtlasTile {
            texture_id: layer_tile_texture_id(key),
            tile_id: layer_tile_id(tile),
            padding: 0,
            bounds: Bounds {
                origin: point(DevicePixels(0), DevicePixels(0)),
                size: device_size(TILE as i32, TILE as i32),
            },
        },
    }
}

/// A frame of `window` that composites `tiles` of `layer` at `translation`,
/// between the panel under the viewport and a scrollbar over it, and, with
/// `path`, a path over both.
fn composited(
    window: &Window,
    layer: LayerFrame,
    tiles: &[TileCoord],
    translation: (f32, f32),
    path: bool,
) -> Scene {
    let mut scene = Scene::default();
    scene.insert_primitive(panel(window));
    scene.push_layer(window.viewport);
    for &tile in tiles {
        scene.insert_primitive(tile_sprite(
            layer.key,
            &layer,
            tile,
            translation,
            window.viewport,
        ));
    }
    scene.pop_layer();
    scene.insert_primitive(scrollbar());
    if path {
        add_path(&mut scene);
    }
    scene.layers.frames.push(layer);
    scene.finish();
    scene
}

/// Draws a window whose viewport shows the layer's content, painted at
/// `origin` in content space and moved by `translation`, once from the
/// content directly and once from the layer's tiles, twice (the second
/// frame from the cached tiles), and compares the pixels.
fn composite_and_compare(
    harness: &mut Harness,
    window: &Window,
    origin: (f32, f32),
    translation: (f32, f32),
    tiles: &[TileCoord],
    path: bool,
) {
    let mut direct = Scene::default();
    direct.insert_primitive(panel(window));
    add_content(
        &mut direct,
        harness,
        (origin.0 + translation.0, origin.1 + translation.1),
        Some(window.viewport),
    );
    direct.insert_primitive(scrollbar());
    if path {
        add_path(&mut direct);
    }
    direct.finish();
    let expected = harness.render(&direct, window.size);
    let width = window.size.width.0 as usize;

    let layer = layer_frame(LayerKey(5), 1, content(harness, origin, None), tiles);
    let scene = composited(window, layer, tiles, translation, path);
    let rasterized = harness.cache().rasterized;
    let actual = harness.render(&scene, window.size);
    assert_eq!(harness.cache().rasterized, rasterized + tiles.len());
    assert_same_pixels(&actual, &expected, width);

    // A frame that only scrolled draws the cached tiles again.
    let actual = harness.render(&scene, window.size);
    assert_eq!(harness.cache().rasterized, rasterized + tiles.len());
    assert_same_pixels(&actual, &expected, width);
}

fn assert_same_pixels(actual: &[u8], expected: &[u8], width: usize) {
    assert_eq!(actual.len(), expected.len());
    let differing: Vec<(usize, usize)> = (0..actual.len() / 4)
        .filter(|i| actual[i * 4..i * 4 + 4] != expected[i * 4..i * 4 + 4])
        .map(|i| (i % width, i / width))
        .collect();
    if differing.is_empty() {
        return;
    }
    let (min_x, max_x) = differing
        .iter()
        .fold((usize::MAX, 0), |(lo, hi), (x, _)| (lo.min(*x), hi.max(*x)));
    let (min_y, max_y) = differing
        .iter()
        .fold((usize::MAX, 0), |(lo, hi), (_, y)| (lo.min(*y), hi.max(*y)));
    panic!(
        "{} pixels differ, within x {min_x}..={max_x}, y {min_y}..={max_y}; first at {:?}: {:?} instead of {:?}",
        differing.len(),
        differing[0],
        pixel(actual, width, differing[0].0, differing[0].1),
        pixel(expected, width, differing[0].0, differing[0].1),
    );
}

/// Content of every primitive kind a layer holds, spread over the four
/// tiles of (0, 0)..(1024, 1024) and across their edges, moved by `offset`
/// and clipped to `clip`.
fn content(harness: &Harness, offset: (f32, f32), clip: Option<Bounds<ScaledPixels>>) -> Scene {
    let mut scene = Scene::default();
    add_content(&mut scene, harness, offset, clip);
    scene.finish();
    scene
}

/// Inserts [`content`]'s primitives into `scene`.
fn add_content(
    scene: &mut Scene,
    harness: &Harness,
    offset: (f32, f32),
    clip: Option<Bounds<ScaledPixels>>,
) {
    let clip = clip.unwrap_or(no_mask().bounds);
    let (ox, oy) = offset;
    let at = |x: f32, y: f32, w: f32, h: f32| sp(x + ox, y + oy, w, h);
    let mask = |bounds: Bounds<ScaledPixels>| ContentMask {
        bounds: bounds.intersect(&clip),
    };

    scene.insert_primitive(Quad {
        content_mask: mask(clip),
        ..quad(at(100., 100., 200., 150.), Hsla::from(rgba(0xcc3322ff)))
    });
    // A bordered, rounded quad across the tile edge at x = 512.
    scene.insert_primitive(Quad {
        bounds: at(450.5, 60., 150., 120.),
        content_mask: mask(at(0., 0., 1024., 1024.)),
        background: Hsla::from(rgba(0xeeeeeeff)).into(),
        border_color: Hsla::from(rgba(0x222222ff)),
        corner_radii: Corners::all(ScaledPixels(12.)),
        border_widths: Edges::all(ScaledPixels(3.)),
        ..Default::default()
    });
    // The shadow's tail crosses the tile edge its bounds stop short of.
    scene.insert_primitive(Shadow {
        order: 0,
        blur_radius: ScaledPixels(8.),
        bounds: at(380., 420., 120., 70.),
        corner_radii: Corners::all(ScaledPixels(6.)),
        content_mask: mask(clip),
        color: Hsla::from(rgba(0x00000080)),
        shape_bounds: at(380., 420., 120., 70.),
        shape_corner_radii: Corners::all(ScaledPixels(6.)),
        inset: false.into(),
        pad: 0,
    });
    scene.insert_primitive(Quad {
        bounds: at(600., 600., 300., 200.),
        content_mask: mask(clip),
        background: linear_gradient(
            45.,
            linear_color_stop(rgba(0xff0000ff), 0.),
            linear_color_stop(rgba(0x0000ffff), 1.),
        ),
        corner_radii: Corners::all(ScaledPixels(20.)),
        ..Default::default()
    });
    scene.insert_primitive(Underline {
        order: 0,
        pad: 0,
        bounds: at(50., 505., 900., 8.),
        content_mask: mask(clip),
        color: Hsla::from(rgba(0xffcc00ff)),
        thickness: ScaledPixels(2.),
        wavy: true.into(),
    });
    scene.insert_primitive(Underline {
        order: 0,
        pad: 0,
        bounds: at(60., 530., 700., 2.),
        content_mask: mask(clip),
        color: Hsla::from(rgba(0x114411ff)),
        thickness: ScaledPixels(1.),
        wavy: false.into(),
    });
    // A clipped quad whose mask crosses a tile edge.
    scene.insert_primitive(Quad {
        bounds: at(700., 200., 200., 200.),
        content_mask: mask(at(490., 250., 300., 100.)),
        background: Hsla::from(rgba(0x44aa44ff)).into(),
        ..Default::default()
    });

    let mono = glyph_tile(harness, 1);
    scene.insert_primitive(MonochromeSprite {
        order: 0,
        animation: Default::default(),
        bounds: at(505., 700., 16., 16.),
        content_mask: mask(clip),
        color: Hsla::from(rgba(0xffffffff)),
        tile: mono,
        transformation: TransformationMatrix::unit(),
    });
    scene.insert_primitive(PolychromeSprite {
        order: 0,
        pad: 0,
        grayscale: false.into(),
        opacity: 1.,
        bounds: at(300., 500., 20., 20.),
        content_mask: mask(clip),
        corner_radii: Corners::all(ScaledPixels(4.)),
        tile: image_tile(harness),
    });
}

/// A path of lines and a curve over the viewport's edge, in window space.
fn add_path(scene: &mut Scene) {
    let mut path = PathBuilder::fill();
    path.move_to(point(px(480.), px(300.)));
    path.line_to(point(px(660.5), px(330.)));
    path.curve_to(point(px(500.), px(460.)), point(px(600.), px(450.)));
    path.line_to(point(px(480.), px(300.)));
    scene.insert_primitive(PaintedPath::new(
        path.build().expect("path"),
        1.,
        ContentMask {
            bounds: no_mask().bounds.map(|c| px(c.0)),
        },
        Hsla::from(rgba(0x8800ffcc)).into(),
    ));
}

/// What GPUI Kit draws for a table's rounded frame: paths filling corner
/// notches, a bordered quad and a glyph drawn over them, and a path nothing
/// covers, moved by `offset` and clipped to `clip`, in drawing order.
fn add_covered_paths(
    scene: &mut Scene,
    harness: &Harness,
    (ox, oy): (f32, f32),
    clip: Bounds<ScaledPixels>,
) {
    let mask = ContentMask { bounds: clip };
    for (x, y, dx, dy) in [
        (200., 330., 1., 1.),
        (330.5, 330., -1., 1.),
        (200., 410.25, 1., -1.),
    ] {
        let corner = point(px(x + ox), px(y + oy));
        let mut path = PathBuilder::fill();
        path.move_to(corner);
        path.line_to(corner + point(px(9. * dx), px(0.)));
        path.curve_to(corner + point(px(0.), px(9. * dy)), corner);
        path.line_to(corner);
        scene.insert_primitive(PaintedPath::new(
            path.build().expect("path"),
            1.,
            ContentMask {
                bounds: clip.map(|c| px(c.0)),
            },
            Hsla::from(rgba(0x3366ccff)).into(),
        ));
    }
    scene.insert_primitive(Quad {
        bounds: sp(200. + ox, 330. + oy, 130.5, 80.25),
        content_mask: mask,
        background: Hsla::from(rgba(0x00000000)).into(),
        border_color: Hsla::from(rgba(0x222222ff)),
        corner_radii: Corners::all(ScaledPixels(9.)),
        border_widths: Edges::all(ScaledPixels(1.)),
        ..Default::default()
    });
    scene.insert_primitive(MonochromeSprite {
        order: 0,
        animation: Default::default(),
        bounds: sp(204. + ox, 334. + oy, 16., 16.),
        content_mask: mask,
        color: Hsla::from(rgba(0xff8800ff)),
        tile: glyph_tile(harness, 1),
        transformation: TransformationMatrix::unit(),
    });
    let mut path = PathBuilder::fill();
    path.move_to(point(px(250. + ox), px(360. + oy)));
    path.line_to(point(px(290.5 + ox), px(372. + oy)));
    path.curve_to(
        point(px(255. + ox), px(400. + oy)),
        point(px(300. + ox), px(395. + oy)),
    );
    path.line_to(point(px(250. + ox), px(360. + oy)));
    scene.insert_primitive(PaintedPath::new(
        path.build().expect("path"),
        1.,
        ContentMask {
            bounds: clip.map(|c| px(c.0)),
        },
        Hsla::from(rgba(0x8800ffcc)).into(),
    ));
}

/// A 16×16 monochrome glyph of a made-up font, uploaded to the renderer's
/// atlas.
fn glyph_tile(harness: &Harness, glyph: u32) -> AtlasTile {
    let key = AtlasKey::Glyph(RenderGlyphParams {
        font_id: FontId(9_999),
        glyph_id: GlyphId(glyph),
        font_size: px(12.),
        subpixel_variant: point(0, 0),
        scale_factor: 1.,
        is_emoji: false,
        subpixel_rendering: false,
        dilation: 0,
    });
    let bytes: Vec<u8> = (0..16 * 16).map(|i| ((i * 37) % 256) as u8).collect();
    harness
        .renderer
        .sprite_atlas()
        .get_or_insert_with(key, &mut || {
            Ok(Some((device_size(16, 16), Cow::Owned(bytes.clone()))))
        })
        .expect("glyph uploaded")
        .expect("glyph tile")
}

/// A 20×20 image uploaded to the renderer's atlas.
fn image_tile(harness: &Harness) -> AtlasTile {
    let key = AtlasKey::Image(RenderImageParams {
        image_id: ImageId(9_999),
        frame_index: 0,
    });
    let bytes: Vec<u8> = (0..20 * 20)
        .flat_map(|i: u32| {
            [
                (i * 7 % 256) as u8,
                (i * 13 % 256) as u8,
                (i * 3 % 256) as u8,
                255,
            ]
        })
        .collect();
    harness
        .renderer
        .sprite_atlas()
        .get_or_insert_with(key, &mut || {
            Ok(Some((device_size(20, 20), Cow::Owned(bytes.clone()))))
        })
        .expect("image uploaded")
        .expect("image tile")
}

// --- helpers --- //

fn sp(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(x), ScaledPixels(y)),
        size: size(ScaledPixels(w), ScaledPixels(h)),
    }
}

/// A content mask that clips nothing the tests draw.
fn no_mask() -> ContentMask<ScaledPixels> {
    ContentMask {
        bounds: sp(-10_000., -10_000., 20_000., 20_000.),
    }
}

fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
    Quad {
        bounds,
        content_mask: no_mask(),
        background: color.into(),
        ..Default::default()
    }
}

fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
    size(DevicePixels(width), DevicePixels(height))
}

/// The RGBA bytes of the pixel at (`x`, `y`) of a `width`-wide readback.
fn pixel(pixels: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let at = (y * width + x) * 4;
    [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
}

/// Draws scenes with a headless renderer, as a window's frames are drawn.
struct Harness {
    renderer: MetalRenderer,
}

impl Harness {
    /// A harness on the system's Metal device, or `None` when there is none
    /// (the tests then skip).
    fn new() -> Option<Harness> {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        match MetalRenderer::new_headless(pool) {
            Ok(renderer) => Some(Harness { renderer }),
            Err(error) => {
                eprintln!("skipped: no Metal device ({error:#})");
                None
            }
        }
    }

    fn cache(&self) -> &TileCache {
        &self.renderer.fast_layers
    }

    /// Draws `scene` into a `size` texture as the renderer draws a frame,
    /// and returns its pixels as RGBA bytes, row by row.
    fn render(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Vec<u8> {
        self.renderer
            .render_scene_to_image(scene, size)
            .expect("scene rendered")
            .into_raw()
    }
}

fn assert_pixels_within(actual: &[u8], expected: &[u8], width: usize, max_delta: u8) {
    assert_eq!(actual.len(), expected.len());
    let mut worst: Option<(usize, usize, [u8; 4], [u8; 4], u8)> = None;
    let mut count = 0usize;
    for i in 0..actual.len() / 4 {
        let a = [
            actual[i * 4],
            actual[i * 4 + 1],
            actual[i * 4 + 2],
            actual[i * 4 + 3],
        ];
        let e = [
            expected[i * 4],
            expected[i * 4 + 1],
            expected[i * 4 + 2],
            expected[i * 4 + 3],
        ];
        let diff = (0..4).map(|c| a[c].abs_diff(e[c])).max().unwrap_or(0);
        if diff > max_delta {
            count += 1;
            if worst.as_ref().is_none_or(|w| diff > w.4) {
                worst = Some((i % width, i / width, a, e, diff));
            }
        }
    }
    if let Some((x, y, a, e, diff)) = worst {
        panic!(
            "{count} pixels differ by more than {max_delta}; worst at ({x}, {y}) delta {diff}: {a:?} instead of {e:?}"
        );
    }
}

fn render_window_fixture(
    cx: &mut gpui::TestAppContext,
    harness: &mut Harness,
    logical_size: Size<gpui::Pixels>,
    build: impl FnMut(&mut gpui::Window, &mut gpui::App) -> gpui::AnyElement + 'static,
) -> (gpui::AnyWindowHandle, Vec<u8>, Size<DevicePixels>) {
    use gpui::{AppContext as _, Context, IntoElement, Render, WindowBounds, WindowOptions};

    struct View(Box<dyn FnMut(&mut gpui::Window, &mut gpui::App) -> gpui::AnyElement>);
    impl Render for View {
        fn render(
            &mut self,
            window: &mut gpui::Window,
            cx: &mut Context<Self>,
        ) -> impl IntoElement {
            (self.0)(window, cx)
        }
    }
    let window = cx.update(|cx| {
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    point(px(0.), px(0.)),
                    logical_size,
                ))),
                ..Default::default()
            },
            |_, cx| cx.new(|_| View(Box::new(build))),
        )
        .unwrap()
    });
    let (pixels, size) = window
        .update(cx, |_, window, cx| {
            window.draw(cx).clear(cx);
            let sf = window.scale_factor();
            let size = device_size(
                (f32::from(logical_size.width) * sf).round() as i32,
                (f32::from(logical_size.height) * sf).round() as i32,
            );
            (harness.render(window.painted_scene(), size), size)
        })
        .unwrap();
    (window.into(), pixels, size)
}

fn redraw_window_fixture(
    cx: &mut gpui::TestAppContext,
    handle: gpui::AnyWindowHandle,
    harness: &mut Harness,
    size: Size<DevicePixels>,
) -> Vec<u8> {
    use gpui::AppContext as _;
    cx.update_window(handle, |_, window, cx| {
        window.refresh();
        window.draw(cx).clear(cx);
        harness.render(window.painted_scene(), size)
    })
    .unwrap()
}

#[gpui::test]
fn cached_layer_opaque_matches_direct_and_skips_hit_rasterization(cx: &mut gpui::TestAppContext) {
    use gpui::{
        CachedLayerId, CachedLayerOptions, IntoElement as _, ParentElement as _, Styled as _,
        canvas, div, fill,
    };
    use std::cell::Cell;

    let mut harness = Harness::new().expect("Metal device required");
    let logical_size = size(px(160.), px(160.));
    let (_, direct_pixels, _) = render_window_fixture(cx, &mut harness, logical_size, |_, _| {
        div()
            .size_full()
            .bg(gpui::white())
            .child(
                canvas(
                    |_, _, _| (),
                    |bounds, (), window, _| {
                        window.paint_quad(fill(
                            Bounds::new(
                                bounds.origin + point(px(16.), px(16.)),
                                size(px(64.), px(64.)),
                            ),
                            Hsla::red(),
                        ));
                    },
                )
                .size_full(),
            )
            .into_any_element()
    });

    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let r0 = harness.cache().rasterized;
    let (handle, pixels1, size_dp) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
            let paints = paints_ref.clone();
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| {
                            window.paint_cached_layer(
                                CachedLayerId::new(1),
                                bounds,
                                1,
                                CachedLayerOptions::default().background(gpui::white()),
                                |window| {
                                    paints.set(paints.get() + 1);
                                    window.paint_quad(fill(
                                        Bounds::new(
                                            bounds.origin + point(px(16.), px(16.)),
                                            size(px(64.), px(64.)),
                                        ),
                                        Hsla::red(),
                                    ));
                                },
                            );
                        },
                    )
                    .size_full(),
                )
                .into_any_element()
        });
    assert_eq!(paints.get(), 1);
    assert_eq!(harness.cache().rasterized, r0 + 1);
    assert_same_pixels(&pixels1, &direct_pixels, size_dp.width.0 as usize);

    // Frame 2 with unchanged revision: neither CPU closure nor GPU tile rasterization runs.
    let pixels2 = redraw_window_fixture(cx, handle, &mut harness, size_dp);
    assert_eq!(paints.get(), 1);
    assert_eq!(harness.cache().rasterized, r0 + 1);
    assert_same_pixels(&pixels2, &direct_pixels, size_dp.width.0 as usize);
}

#[test]
fn cached_layer_transparent_matches_direct_over_gradient_backdrop() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipped: no Metal device");
        return;
    };
    let size = device_size(256, 256);
    let viewport = sp(0., 0., 256., 256.);
    let key = LayerKey(50);
    let glyph = glyph_tile(&harness, 0);

    let build_foreground = |scene: &mut Scene, offset: (f32, f32)| {
        let mask = ContentMask { bounds: viewport };
        scene.insert_primitive(Quad {
            order: 0,
            background: rgba(0x112233ff).into(),
            bounds: sp(offset.0 + 16., offset.1 + 16., 48., 48.),
            content_mask: mask,
            ..Default::default()
        });
        scene.insert_primitive(Quad {
            order: 0,
            background: rgba(0xee224480).into(),
            bounds: sp(offset.0 + 48., offset.1 + 48., 110., 110.),
            content_mask: mask,
            corner_radii: Corners::all(ScaledPixels(12.)),
            ..Default::default()
        });
        scene.insert_primitive(Quad {
            order: 0,
            background: rgba(0x22ee6680).into(),
            bounds: sp(offset.0 + 96., offset.1 + 88., 96., 96.),
            content_mask: mask,
            ..Default::default()
        });
        scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(8.),
            bounds: sp(offset.0 + 40., offset.1 + 180., 140., 40.),
            corner_radii: Corners::all(ScaledPixels(8.)),
            content_mask: mask,
            color: Hsla::from(rgba(0x00000090)),
            shape_bounds: sp(offset.0 + 40., offset.1 + 180., 140., 40.),
            shape_corner_radii: Corners::all(ScaledPixels(8.)),
            inset: false.into(),
            pad: 0,
        });
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            animation: Default::default(),
            bounds: sp(offset.0 + 170., offset.1 + 30., 24., 24.),
            content_mask: mask,
            color: Hsla::from(rgba(0xffffffc0)),
            tile: glyph.clone(),
            transformation: TransformationMatrix::unit(),
        });
    };

    let backdrop = || Quad {
        bounds: viewport,
        content_mask: ContentMask { bounds: viewport },
        background: linear_gradient(
            90.,
            linear_color_stop(rgba(0x1144aaff), 0.),
            linear_color_stop(rgba(0xff6622ff), 1.),
        ),
        ..Default::default()
    };

    let mut direct = Scene::default();
    direct.insert_primitive(backdrop());
    build_foreground(&mut direct, (0., 0.));
    direct.finish();
    let expected = harness.render(&direct, size);

    let mut content = Scene::default();
    build_foreground(&mut content, (0., 0.));
    content.finish();

    let layer = LayerFrame {
        key,
        generation: 1,
        background: Rgba {
            r: 0.,
            g: 0.,
            b: 0.,
            a: 0.,
        },
        tile_size: TILE,
        content: Rc::new(content).into(),
        dirty_tiles: vec![coord(0, 0)],
    };
    let mut cached = Scene::default();
    cached.insert_primitive(backdrop());
    cached.push_layer(viewport);
    cached.insert_primitive(tile_sprite(key, &layer, coord(0, 0), (0., 0.), viewport));
    cached.pop_layer();
    cached.layers.frames.push(layer);
    cached.finish();

    let r0 = harness.cache().rasterized;
    let actual = harness.render(&cached, size);
    assert!(harness.cache().is_premultiplied(key));
    assert_eq!(harness.cache().rasterized, r0 + 1);
    assert_pixels_within(&actual, &expected, size.width.0 as usize, 2);

    // Opaque quad and transparent corners match byte-for-byte with zero halo.
    assert_eq!(
        pixel(&actual, size.width.0 as usize, 32, 32),
        pixel(&expected, size.width.0 as usize, 32, 32)
    );
    assert_eq!(
        pixel(&actual, size.width.0 as usize, 240, 240),
        pixel(&expected, size.width.0 as usize, 240, 240)
    );

    // Cache hit does not re-rasterize.
    let actual2 = harness.render(&cached, size);
    assert_eq!(harness.cache().rasterized, r0 + 1);
    assert_eq!(actual, actual2);
}

#[gpui::test]
fn cached_layer_transparent_inside_fast_rounded_clip_matches_direct(cx: &mut gpui::TestAppContext) {
    use gpui::{
        CachedLayerId, CachedLayerOptions, IntoElement as _, ParentElement as _, Styled as _,
        canvas, div, fill,
    };

    let mut harness = Harness::new().expect("Metal device required");
    let logical_size = size(px(128.), px(128.));
    let make_div = |cached: bool| {
        let backdrop = linear_gradient(
            45.,
            linear_color_stop(rgba(0x183060ff), 0.),
            linear_color_stop(rgba(0xd05020ff), 1.),
        );
        div()
            .size_full()
            .bg(backdrop)
            .child(
                div().size_full().rounded(px(28.)).overflow_hidden().child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| {
                            let paint = |window: &mut gpui::Window| {
                                window.paint_quad(fill(
                                    Bounds::new(bounds.origin, size(px(128.), px(64.))),
                                    rgba(0xee335599),
                                ));
                                window.paint_quad(fill(
                                    Bounds::new(
                                        bounds.origin + point(px(20.), px(20.)),
                                        size(px(88.), px(88.)),
                                    ),
                                    rgba(0x22cc8888),
                                ));
                            };
                            if cached {
                                window.paint_cached_layer(
                                    CachedLayerId::new(99),
                                    bounds,
                                    1,
                                    CachedLayerOptions::default().background(Rgba {
                                        r: 0.,
                                        g: 0.,
                                        b: 0.,
                                        a: 0.,
                                    }),
                                    paint,
                                );
                            } else {
                                paint(window);
                            }
                        },
                    )
                    .size_full(),
                ),
            )
            .into_any_element()
    };

    let (_, expected, size_dp) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| make_div(false));
    let (_, actual, _) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| make_div(true));

    assert_pixels_within(&actual, &expected, size_dp.width.0 as usize, 2);
    // Clipped corner pixel must match backdrop gradient untouched.
    assert_eq!(
        pixel(&actual, size_dp.width.0 as usize, 2, 2),
        pixel(&expected, size_dp.width.0 as usize, 2, 2)
    );
}

#[gpui::test]
fn cached_layer_multi_tile_partial_update_rasterizes_only_dirty_tile(
    cx: &mut gpui::TestAppContext,
) {
    use gpui::{
        CachedLayerId, CachedLayerOptions, IntoElement as _, ParentElement as _, Styled as _,
        canvas, div, fill,
    };
    use std::cell::Cell;

    let mut harness = Harness::new().expect("Metal device required");
    let logical_size = size(px(800.), px(200.));
    let draw_pair = |window: &mut gpui::Window, bounds: Bounds<gpui::Pixels>, right_color: Hsla| {
        window.paint_quad(fill(
            Bounds::new(
                bounds.origin + point(px(20.), px(20.)),
                size(px(120.), px(120.)),
            ),
            Hsla::red(),
        ));
        window.paint_quad(fill(
            Bounds::new(
                bounds.origin + point(px(580.), px(20.)),
                size(px(120.), px(120.)),
            ),
            right_color,
        ));
    };

    let (_, expected1, size_dp) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| draw_pair(window, bounds, Hsla::blue()),
                    )
                    .size_full(),
                )
                .into_any_element()
        });
    let (_, expected2, _) = render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
        div()
            .size_full()
            .bg(gpui::white())
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, (), window, _| draw_pair(window, bounds, Hsla::green()),
                )
                .size_full(),
            )
            .into_any_element()
    });

    let rev = Rc::new(Cell::new(1u64));
    let rev_ref = rev.clone();
    let r0 = harness.cache().rasterized;
    let (handle, actual1, _) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
            let r = rev_ref.get();
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| {
                            window.paint_cached_layer(
                                CachedLayerId::new(77),
                                bounds,
                                r,
                                CachedLayerOptions::default().background(gpui::white()),
                                |window| {
                                    let right = if r == 1 { Hsla::blue() } else { Hsla::green() };
                                    draw_pair(window, bounds, right);
                                },
                            );
                        },
                    )
                    .size_full(),
                )
                .into_any_element()
        });

    let cols = (size_dp.width.0 as u32).div_ceil(TILE) as usize;
    let rows = (size_dp.height.0 as u32).div_ceil(TILE) as usize;
    let tile_count = cols * rows;
    assert!(tile_count >= 2);
    assert_eq!(harness.cache().rasterized, r0 + tile_count);
    assert_same_pixels(&actual1, &expected1, size_dp.width.0 as usize);

    // Bump revision changing only the right-hand tile (`x = 580`): only 1 tile rasterizes!
    rev.set(2);
    let actual2 = redraw_window_fixture(cx, handle, &mut harness, size_dp);
    assert_eq!(harness.cache().rasterized, r0 + tile_count + 1);
    assert_same_pixels(&actual2, &expected2, size_dp.width.0 as usize);
}

/// A window may draw frames its renderer never renders (no drawable while
/// the window is in the background, or a frame drawn outside the display
/// link's update). A cached layer repainted in such a frame must still show
/// its new content once a later frame is rendered: the dirty tiles belong to
/// the generation, not to the frame that made it.
#[gpui::test]
fn cached_layer_repainted_in_an_unrendered_frame_shows_its_new_content(
    cx: &mut gpui::TestAppContext,
) {
    use gpui::{
        AppContext as _, CachedLayerId, CachedLayerOptions, IntoElement as _, ParentElement as _,
        Styled as _, canvas, div, fill,
    };
    use std::cell::Cell;

    let logical_size = size(px(240.), px(160.));
    let paint = |window: &mut gpui::Window, bounds: Bounds<gpui::Pixels>, color: Hsla| {
        window.paint_quad(fill(
            Bounds::new(
                bounds.origin + point(px(20.), px(20.)),
                size(px(120.), px(80.)),
            ),
            color,
        ));
    };
    let direct = |color: Hsla| {
        move |_: &mut gpui::Window, _: &mut gpui::App| {
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| paint(window, bounds, color),
                    )
                    .size_full(),
                )
                .into_any_element()
        }
    };

    // `revisions` lists the revision of each frame and whether it is
    // rendered; the content is blue at revision 1 and green after.
    for (what, revisions) in [
        ("reused", [(1u64, true), (2, false), (2, true)]),
        (
            "identically repainted",
            [(1u64, true), (2, false), (3, true)],
        ),
    ] {
        let mut harness = Harness::new().expect("Metal device required");
        let (_, green, size_dp) =
            render_window_fixture(cx, &mut harness, logical_size, direct(Hsla::green()));
        let mut harness = Harness::new().expect("Metal device required");
        let rev = Rc::new(Cell::new(revisions[0].0));
        let rev_ref = rev.clone();
        let (handle, _, _) = render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
            let r = rev_ref.get();
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| {
                            window.paint_cached_layer(
                                CachedLayerId::new(5),
                                bounds,
                                r,
                                CachedLayerOptions::default().background(gpui::white()),
                                |window| {
                                    let color = if r == 1 { Hsla::blue() } else { Hsla::green() };
                                    paint(window, bounds, color);
                                },
                            );
                        },
                    )
                    .size_full(),
                )
                .into_any_element()
        });
        let mut last: Option<Vec<u8>> = None;
        for &(revision, rendered) in &revisions[1..] {
            rev.set(revision);
            last = cx
                .update_window(handle, |_, window, cx| {
                    window.refresh();
                    window.draw(cx).clear(cx);
                    rendered.then(|| harness.render(window.painted_scene(), size_dp))
                })
                .unwrap();
        }
        let actual = last.expect("the last frame is rendered");
        let width = size_dp.width.0 as usize;
        let at = (40.0 * size_dp.width.0 as f32 / 240.0) as usize;
        assert_eq!(
            pixel(&actual, width, at, at),
            pixel(&green, width, at, at),
            "{what}: the layer still shows what it held before the unrendered frame"
        );
        assert_same_pixels(&actual, &green, width);
    }
}

/// A cached layer painted again over smaller bounds shows nothing outside
/// them. The tiles it no longer covers still hold what it painted there
/// before: a table scrolled sideways showed its copy from before the scroll
/// under the new one, cut at a tile edge.
#[gpui::test]
fn cached_layer_painted_again_over_smaller_bounds_shows_nothing_outside_them(
    cx: &mut gpui::TestAppContext,
) {
    use gpui::{
        CachedLayerId, CachedLayerOptions, IntoElement as _, ParentElement as _, Styled as _,
        canvas, div, fill,
    };
    use std::cell::Cell;

    let logical_size = size(px(200.), px(600.));
    let rect = |window: &mut gpui::Window,
                origin: gpui::Point<gpui::Pixels>,
                top: f32,
                height: f32,
                color: Hsla| {
        window.paint_quad(fill(
            Bounds::new(origin + point(px(20.), px(top)), size(px(160.), px(height))),
            color,
        ));
    };
    // Drawn under the layer, the band leaves no solid colour to bake under
    // it: its tiles are transparent, as over a window's backdrop image.
    let band = move |window: &mut gpui::Window, bounds: Bounds<gpui::Pixels>| {
        window.paint_quad(fill(
            Bounds::new(
                bounds.origin + point(px(0.), px(50.)),
                size(bounds.size.width, px(550.)),
            ),
            gpui::hsla(0., 0., 0.5, 1.),
        ));
    };

    let mut harness = Harness::new().expect("Metal device required");
    let (_, expected, size_dp) =
        render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
            div()
                .size_full()
                .bg(gpui::white())
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, (), window, _| {
                            band(window, bounds);
                            rect(window, bounds.origin, 20., 40., Hsla::blue());
                        },
                    )
                    .size_full(),
                )
                .into_any_element()
        });

    // First the whole height with a block at the bottom, a tile row below
    // the top one, then only the top 100 px with a block in them.
    let tall = Rc::new(Cell::new(true));
    let tall_ref = tall.clone();
    let mut harness = Harness::new().expect("Metal device required");
    let (handle, _, _) = render_window_fixture(cx, &mut harness, logical_size, move |_, _| {
        let tall = tall_ref.get();
        div()
            .size_full()
            .bg(gpui::white())
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, (), window, _| {
                        band(window, bounds);
                        let layer = if tall {
                            bounds
                        } else {
                            Bounds::new(bounds.origin, size(bounds.size.width, px(100.)))
                        };
                        window.paint_cached_layer(
                            CachedLayerId::new(9),
                            layer,
                            if tall { 1 } else { 2 },
                            CachedLayerOptions::default(),
                            |window| {
                                if tall {
                                    rect(window, bounds.origin, 540., 50., Hsla::blue());
                                } else {
                                    rect(window, bounds.origin, 20., 40., Hsla::blue());
                                }
                            },
                        );
                    },
                )
                .size_full(),
            )
            .into_any_element()
    });
    tall.set(false);
    let actual = redraw_window_fixture(cx, handle, &mut harness, size_dp);

    let width = size_dp.width.0 as usize;
    let scale = size_dp.width.0 as f32 / 200.;
    let (x, y) = ((100. * scale) as usize, (565. * scale) as usize);
    assert_eq!(
        pixel(&actual, width, x, y),
        pixel(&expected, width, x, y),
        "the block painted before the layer shrank still shows below it"
    );
    assert_pixels_within(&actual, &expected, width, 2);
}

#[path = "../../../../gpui_linux/src/fast/composition/occluders.rs"]
mod x11_occluder_tests;
