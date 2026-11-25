// Modified by Longbridge for gpui-fast.
//! Drawing scroll layer tiles where the scene composites them: polychrome
//! sprites whose texture id lies in the range no atlas allocates stand for
//! tiles, and are drawn with the polychrome sprite pipeline from the tiles'
//! own textures instead of an atlas texture.
//!
//! A tile sprite covers the tile's whole texture at whole device pixels, so
//! every fragment samples a texel center and copies the tile's pixel.

use std::ops::Range;

use gpui::{
    AtlasTextureId, LayerKey, PolychromeSprite, Scene, TileCoord, TileId, decode_layer_tile,
};

use crate::fast::layers::TileCache;
use crate::wgpu_renderer::{DEPTH_FORMAT, WgpuBindGroupLayouts};

/// Builds the premultiplied-alpha render pipeline used when compositing
/// layer tiles whose background is transparent or translucent.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    layouts: &WgpuBindGroupLayouts,
    surface_format: wgpu::TextureFormat,
    shader_module: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let mut bind_group_layouts = vec![
        Some(&layouts.globals),
        Some(&layouts.instances),
        Some(&layouts.texture),
    ];
    crate::fast::rounded_clip::pipeline_layouts(&mut bind_group_layouts, layouts);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fast_layer_tiles_layout"),
        bind_group_layouts: &bind_group_layouts,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fast_layer_tiles"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: Some("vs_poly_sprite"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: Some("fs_layer_tile"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The layer tiles `scene` composites, in the order it lists them.
pub(crate) fn composited_tiles(scene: &Scene) -> impl Iterator<Item = (LayerKey, TileCoord)> + '_ {
    let has_layers = !scene.layers.frames.is_empty();
    let direct = scene
        .polychrome_sprites
        .iter()
        .take_while(move |_| has_layers)
        .filter_map(|sprite| decode_layer_tile(sprite.tile.texture_id, sprite.tile.tile_id));
    if !scene
        .polychrome_sprites
        .iter()
        .any(|sprite| crate::fast::rounded_clip::group(scene, sprite.tile.texture_id).is_some())
    {
        return itertools::Either::Left(direct);
    }
    let mut tiles: Vec<_> = direct.collect();
    for sprite in &scene.polychrome_sprites {
        if let Some(child) = crate::fast::rounded_clip::group(scene, sprite.tile.texture_id)
            .and_then(|frame| frame.content.scene())
        {
            tiles.extend(composited_tiles(child));
        }
    }
    itertools::Either::Right(tiles.into_iter())
}

/// Whether sprites of `texture` are layer tiles rather than atlas sprites.
pub(crate) fn is_layer_texture(texture: AtlasTextureId) -> bool {
    decode_layer_tile(texture, TileId(0)).is_some()
}

/// Whether `texture` belongs to a layer cleared with a non-opaque background.
pub(crate) fn is_premultiplied_layer(cache: &TileCache, texture: AtlasTextureId) -> bool {
    decode_layer_tile(texture, TileId(0)).is_some_and(|(layer, _)| cache.is_premultiplied(layer))
}

/// Splits a batch of layer tile sprites into runs of the same tile: the
/// scene batches sprites by texture id, which is one per layer, while each
/// tile has a texture of its own.
pub(crate) fn tile_runs(
    sprites: &[PolychromeSprite],
    range: Range<usize>,
) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = range.start;
    std::iter::from_fn(move || {
        if start >= range.end {
            return None;
        }
        let tile = sprites[start].tile.tile_id;
        let end = (start..range.end)
            .find(|&index| sprites[index].tile.tile_id != tile)
            .unwrap_or(range.end);
        let run = start..end;
        start = end;
        Some(run)
    })
}

/// The texture a run of sprites of one layer tile is drawn from, or `None`
/// when the cache does not hold the tile, which rasterizing every composited
/// tile it lacks before the frame is drawn prevents.
pub(crate) fn texture_for_batch<'a>(
    cache: &'a TileCache,
    texture: AtlasTextureId,
    sprites: &[PolychromeSprite],
) -> Option<&'a wgpu::TextureView> {
    let first = sprites.first()?;
    debug_assert!(
        sprites
            .iter()
            .all(|sprite| sprite.tile.tile_id == first.tile.tile_id),
        "a run of layer tile sprites spans tiles"
    );
    let (layer, tile) = decode_layer_tile(texture, first.tile.tile_id)?;
    let view = cache.texture(layer, tile);
    debug_assert!(view.is_some(), "a composited layer tile was not rasterized");
    view
}
