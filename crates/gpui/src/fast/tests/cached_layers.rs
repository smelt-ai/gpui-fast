use crate::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Bounds, CachedLayerId, CachedLayerOptions,
    ContentMask, Context, DevicePixels, Entity, Hsla, PathBuilder, Pixels, Point, Render, Rgba,
    ScaledPixels, SubpixelSprite, TileCoord, TileId, TransformationMatrix, Window, canvas,
    decode_layer_tile, div, fill, point, prelude::*, px, size,
};
use std::cell::Cell;
use std::rc::Rc;

struct CachedCanvasView {
    bounds: Bounds<Pixels>,
    revision: u64,
    options: CachedLayerOptions,
    paints: Rc<Cell<usize>>,
    paint_fn: Rc<dyn Fn(&mut Window, &mut Context<Self>)>,
}

impl Render for CachedCanvasView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bounds = self.bounds;
        let revision = self.revision;
        let options = self.options;
        let paints = self.paints.clone();
        let paint_fn = self.paint_fn.clone();
        let entity = cx.entity();
        div().size_full().child(
            canvas(
                |_, _, _| (),
                move |_, (), window, cx| {
                    entity.update(cx, |this, cx| {
                        let entity_bounds = this.bounds;
                        window.paint_cached_layer(
                            CachedLayerId::new(1),
                            bounds.union(&entity_bounds),
                            revision,
                            options,
                            |window| {
                                paints.set(paints.get() + 1);
                                paint_fn(window, cx);
                            },
                        );
                    });
                },
            )
            .size_full(),
        )
    }
}

fn draw_window<V: 'static>(cx: &mut crate::TestAppContext, handle: crate::WindowHandle<V>) {
    cx.update_window(handle.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
}

#[crate::test]
fn cached_layer_revision_hit_reuses_content_and_keeps_its_dirty_tiles(
    cx: &mut crate::TestAppContext,
) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let handle = cx.add_window(|_, _| CachedCanvasView {
        bounds: Bounds::new(point(px(20.), px(20.)), size(px(120.), px(80.))),
        revision: 1,
        options: CachedLayerOptions::default().background(crate::white()),
        paints: paints_ref,
        paint_fn: Rc::new(|window, _| {
            window.paint_quad(fill(
                Bounds::new(point(px(30.), px(30.)), size(px(40.), px(40.))),
                Hsla::red(),
            ));
        }),
    });

    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    let (gen1, dirty1) = handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert!(!frames[0].dirty_tiles.is_empty());
            (frames[0].generation, frames[0].dirty_tiles.clone())
        })
        .unwrap();

    // Redraw with unchanged revision: `paint` closure stays skipped, and the
    // frame describes the same generation with the same dirty tiles (the
    // renderer may not have rendered the frame that made it).
    handle.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].generation, gen1);
            assert_eq!(frames[0].dirty_tiles, dirty1);
        })
        .unwrap();

    // Bump revision with modified content: repaints and advances generation.
    handle
        .update(cx, |view, _, cx| {
            view.revision = 2;
            view.paint_fn = Rc::new(|window, _| {
                window.paint_quad(fill(
                    Bounds::new(point(px(30.), px(30.)), size(px(40.), px(40.))),
                    Hsla::blue(),
                ));
            });
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 2);
    handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert!(frames[0].generation > gen1);
            assert!(!frames[0].dirty_tiles.is_empty());
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_multi_tile_partial_dirtying(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let handle = cx.add_window(|_, _| CachedCanvasView {
        bounds: Bounds::new(point(px(0.), px(0.)), size(px(900.), px(300.))),
        revision: 1,
        options: CachedLayerOptions::default().background(crate::white()),
        paints: paints_ref,
        paint_fn: Rc::new(|window, _| {
            window.paint_quad(fill(
                Bounds::new(point(px(20.), px(20.)), size(px(80.), px(80.))),
                Hsla::red(),
            ));
            window.paint_quad(fill(
                Bounds::new(point(px(600.), px(20.)), size(px(80.), px(80.))),
                Hsla::blue(),
            ));
        }),
    });

    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    let sf = handle
        .update(cx, |_, window, _| window.scale_factor())
        .unwrap();
    let expected_cols = ((900. * sf).ceil() as u32).div_ceil(512) as i32;
    let expected_rows = ((300. * sf).ceil() as u32).div_ceil(512) as i32;
    let gen1 = handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert_eq!(
                frames[0].dirty_tiles.len(),
                (expected_cols * expected_rows) as usize
            );
            frames[0].generation
        })
        .unwrap();

    // Update only the right tile (`x = 600` logical px): only that tile dirties.
    handle
        .update(cx, |view, _, cx| {
            view.revision = 2;
            view.paint_fn = Rc::new(|window, _| {
                window.paint_quad(fill(
                    Bounds::new(point(px(20.), px(20.)), size(px(80.), px(80.))),
                    Hsla::red(),
                ));
                window.paint_quad(fill(
                    Bounds::new(point(px(600.), px(20.)), size(px(80.), px(80.))),
                    Hsla::green(),
                ));
            });
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 2);
    let dirty_col = (600. * sf / 512.).floor() as i32;
    handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].generation, gen1 + 1);
            assert_eq!(
                frames[0].dirty_tiles,
                vec![TileCoord { x: dirty_col, y: 0 }]
            );
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_identical_repaint_keeps_generation(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let paint_quad: Rc<dyn Fn(&mut Window, &mut Context<CachedCanvasView>)> =
        Rc::new(|window, _| {
            window.paint_quad(fill(
                Bounds::new(point(px(16.), px(16.)), size(px(100.), px(100.))),
                Hsla::blue(),
            ));
        });
    let handle = cx.add_window(|_, _| CachedCanvasView {
        bounds: Bounds::new(point(px(10.), px(10.)), size(px(200.), px(200.))),
        revision: 1,
        options: CachedLayerOptions::default().background(crate::white()),
        paints: paints_ref,
        paint_fn: paint_quad,
    });

    draw_window(cx, handle);
    let (gen1, dirty1) = handle
        .update(cx, |_, window, _| {
            let frame = &window.rendered_frame.scene.layers.frames[0];
            (frame.generation, frame.dirty_tiles.clone())
        })
        .unwrap();

    handle
        .update(cx, |view, _, cx| {
            view.revision = 2;
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 2);
    handle
        .update(cx, |_, window, _| {
            let frames = &window.rendered_frame.scene.layers.frames;
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].generation, gen1);
            assert_eq!(frames[0].dirty_tiles, dirty1);
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_translation_and_offset_without_repaint(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct TranslatedView {
        bounds: Bounds<Pixels>,
        offset: Point<Pixels>,
        paints: Rc<Cell<usize>>,
    }
    impl Render for TranslatedView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let bounds = self.bounds;
            let offset = self.offset;
            let paints = self.paints.clone();
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        window.paint_cached_layer(
                            CachedLayerId::new(7),
                            bounds,
                            1,
                            CachedLayerOptions::default()
                                .background(crate::white())
                                .offset(offset),
                            |window| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(fill(
                                    Bounds::new(
                                        bounds.origin + offset + point(px(10.), px(10.)),
                                        size(px(40.), px(40.)),
                                    ),
                                    Hsla::red(),
                                ));
                            },
                        );
                    },
                )
                .size_full(),
            )
        }
    }

    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let handle = cx.add_window(|_, _| TranslatedView {
        bounds: Bounds::new(point(px(20.), px(30.)), size(px(120.), px(120.))),
        offset: Point::default(),
        paints: paints_ref,
    });

    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    let (gen1, origin1) = handle
        .update(cx, |_, window, _| {
            let frame = &window.rendered_frame.scene.layers.frames[0];
            let sprite = window
                .rendered_frame
                .scene
                .polychrome_sprites
                .iter()
                .find(|s| {
                    decode_layer_tile(s.tile.texture_id, s.tile.tile_id)
                        .is_some_and(|(_, tile)| tile == TileCoord { x: 0, y: 0 })
                })
                .unwrap();
            (frame.generation, sprite.bounds.origin)
        })
        .unwrap();

    // Move bounds.origin: cache hit, shifted sprite origin, no repaint.
    handle
        .update(cx, |view, _, cx| {
            view.bounds.origin = point(px(65.), px(50.));
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    let origin2 = handle
        .update(cx, |_, window, _| {
            let frame = &window.rendered_frame.scene.layers.frames[0];
            assert_eq!(frame.generation, gen1);
            let sprite = window
                .rendered_frame
                .scene
                .polychrome_sprites
                .iter()
                .find(|s| {
                    decode_layer_tile(s.tile.texture_id, s.tile.tile_id)
                        .is_some_and(|(_, tile)| tile == TileCoord { x: 0, y: 0 })
                })
                .unwrap();
            sprite.bounds.origin
        })
        .unwrap();
    let sf = handle
        .update(cx, |_, window, _| window.scale_factor())
        .unwrap();
    assert_eq!(
        origin2,
        point(
            origin1.x + ScaledPixels(45. * sf),
            origin1.y + ScaledPixels(20. * sf)
        )
    );

    // Move options.offset: cache hit, shifted sprite origin again.
    handle
        .update(cx, |view, _, cx| {
            view.offset = point(px(12.), px(-8.));
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    let origin3 = handle
        .update(cx, |_, window, _| {
            let sprite = window
                .rendered_frame
                .scene
                .polychrome_sprites
                .iter()
                .find(|s| {
                    decode_layer_tile(s.tile.texture_id, s.tile.tile_id)
                        .is_some_and(|(_, tile)| tile == TileCoord { x: 0, y: 0 })
                })
                .unwrap();
            sprite.bounds.origin
        })
        .unwrap();
    assert_eq!(
        origin3,
        point(
            origin2.x + ScaledPixels(12. * sf),
            origin2.y - ScaledPixels(8. * sf)
        )
    );
}

#[crate::test]
fn cached_layer_remove_and_ttl_eviction(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct TwoLayers {
        paint_second: bool,
    }
    impl Render for TwoLayers {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paint_second = self.paint_second;
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(10.), px(10.)), size(px(80.), px(80.)));
                        window.paint_cached_layer(
                            CachedLayerId::new(1),
                            bounds,
                            1,
                            CachedLayerOptions::default(),
                            |window| {
                                window.paint_quad(fill(bounds, Hsla::red()));
                            },
                        );
                        if paint_second {
                            window.paint_cached_layer(
                                CachedLayerId::new(2),
                                bounds,
                                1,
                                CachedLayerOptions::default(),
                                |window| {
                                    window.paint_quad(fill(bounds, Hsla::blue()));
                                },
                            );
                        }
                    },
                )
                .size_full(),
            )
        }
    }

    let handle = cx.add_window(|_, _| TwoLayers { paint_second: true });
    draw_window(cx, handle);
    let key1_before = handle
        .update(cx, |_, window, _| {
            assert!(
                window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(1))
            );
            assert!(
                window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(2))
            );
            window.fast_layers.cached.entries[&CachedLayerId::new(1)].key
        })
        .unwrap();

    // Explicit removal clears immediately; next paint allocates a fresh LayerKey.
    handle
        .update(cx, |view, window, cx| {
            window.remove_cached_layer(CachedLayerId::new(1));
            assert!(
                !window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(1))
            );
            view.paint_second = false;
            cx.notify();
        })
        .unwrap();
    draw_window(cx, handle);
    let key1_after = handle
        .update(cx, |_, window, _| {
            window.fast_layers.cached.entries[&CachedLayerId::new(1)].key
        })
        .unwrap();
    assert_ne!(key1_before, key1_after);

    // Advance DROP_AFTER_FRAMES without painting CachedLayerId(2): TTL evicts it cleanly.
    for _ in 0..=crate::fast::layers::policy::DROP_AFTER_FRAMES {
        handle.update(cx, |_, _, cx| cx.notify()).unwrap();
        draw_window(cx, handle);
    }
    handle
        .update(cx, |_, window, _| {
            assert!(
                !window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(2))
            );
            assert!(
                window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(1))
            );
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_nested_call_bypasses_caching(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct NestedView {
        outer_paints: Rc<Cell<usize>>,
        inner_paints: Rc<Cell<usize>>,
    }
    impl Render for NestedView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let outer_paints = self.outer_paints.clone();
            let inner_paints = self.inner_paints.clone();
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(160.), px(160.)));
                        window.paint_cached_layer(
                            CachedLayerId::new(10),
                            bounds,
                            1,
                            CachedLayerOptions::default().background(crate::white()),
                            |window| {
                                outer_paints.set(outer_paints.get() + 1);
                                let ran_inner = window.paint_cached_layer(
                                    CachedLayerId::new(11),
                                    Bounds::new(point(px(10.), px(10.)), size(px(60.), px(60.))),
                                    1,
                                    CachedLayerOptions::default(),
                                    |window| {
                                        inner_paints.set(inner_paints.get() + 1);
                                        window.paint_quad(fill(
                                            Bounds::new(
                                                point(px(15.), px(15.)),
                                                size(px(30.), px(30.)),
                                            ),
                                            Hsla::green(),
                                        ));
                                    },
                                );
                                assert!(ran_inner.is_some());
                            },
                        );
                    },
                )
                .size_full(),
            )
        }
    }

    let outer_paints = Rc::new(Cell::new(0usize));
    let inner_paints = Rc::new(Cell::new(0usize));
    let handle = cx.add_window(|_, _| NestedView {
        outer_paints: outer_paints.clone(),
        inner_paints: inner_paints.clone(),
    });
    draw_window(cx, handle);
    assert_eq!(outer_paints.get(), 1);
    assert_eq!(inner_paints.get(), 1);
    handle
        .update(cx, |_, window, _| {
            assert_eq!(window.rendered_frame.scene.layers.frames.len(), 1);
            assert!(
                !window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(11))
            );
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_transparent_subpixel_falls_back_to_direct_with_overlay(
    cx: &mut crate::TestAppContext,
) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct SubpixelWithPathView {
        paints: Rc<Cell<usize>>,
    }
    impl Render for SubpixelWithPathView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(200.), px(200.)));
                        window.paint_cached_layer(
                            CachedLayerId::new(20),
                            bounds,
                            1,
                            CachedLayerOptions::default().background(Rgba {
                                r: 0.,
                                g: 0.,
                                b: 0.,
                                a: 0.,
                            }),
                            |window| {
                                paints.set(paints.get() + 1);
                                window.next_frame.scene.insert_primitive(SubpixelSprite {
                                    order: 0,
                                    pad: 0,
                                    tile: AtlasTile {
                                        texture_id: AtlasTextureId {
                                            index: 0,
                                            kind: AtlasTextureKind::Subpixel,
                                        },
                                        tile_id: TileId(1),
                                        padding: 0,
                                        bounds: Bounds::new(
                                            point(DevicePixels(0), DevicePixels(0)),
                                            size(DevicePixels(8), DevicePixels(8)),
                                        ),
                                    },
                                    bounds: Bounds::new(
                                        point(ScaledPixels(10.), ScaledPixels(10.)),
                                        size(ScaledPixels(8.), ScaledPixels(8.)),
                                    ),
                                    content_mask: ContentMask {
                                        bounds: Bounds::new(
                                            point(ScaledPixels(0.), ScaledPixels(0.)),
                                            size(ScaledPixels(200.), ScaledPixels(200.)),
                                        ),
                                    },
                                    color: Hsla::black(),
                                    transformation: TransformationMatrix::unit(),
                                });
                                let mut path = PathBuilder::fill();
                                path.move_to(point(px(40.), px(40.)));
                                path.line_to(point(px(80.), px(40.)));
                                path.line_to(point(px(60.), px(80.)));
                                path.close();
                                window.paint_path(path.build().unwrap(), Hsla::red());
                            },
                        );
                    },
                )
                .size_full(),
            )
        }
    }

    let paints = Rc::new(Cell::new(0usize));
    let handle = cx.add_window(|_, _| SubpixelWithPathView {
        paints: paints.clone(),
    });
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    handle
        .update(cx, |_, window, _| {
            assert!(window.rendered_frame.scene.layers.frames.is_empty());
            assert_eq!(window.rendered_frame.scene.subpixel_sprites.len(), 1);
            assert_eq!(window.rendered_frame.scene.paths.len(), 1);
        })
        .unwrap();

    // Revision hit still skips `paint` while preserving direct subpixel + path overlay.
    handle.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    handle
        .update(cx, |_, window, _| {
            assert!(window.rendered_frame.scene.layers.frames.is_empty());
            assert_eq!(window.rendered_frame.scene.subpixel_sprites.len(), 1);
            assert_eq!(window.rendered_frame.scene.paths.len(), 1);
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_retained_view_replay_keeps_layer_frame(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct ChildView {
        paints: Rc<Cell<usize>>,
    }
    impl Render for ChildView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let paints = self.paints.clone();
            div().size(px(100.)).child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(100.), px(100.)));
                        window.paint_cached_layer(
                            CachedLayerId::new(30),
                            bounds,
                            1,
                            CachedLayerOptions::default().background(crate::white()),
                            |window| {
                                paints.set(paints.get() + 1);
                                window.paint_quad(fill(bounds, Hsla::blue()));
                            },
                        );
                    },
                )
                .size_full(),
            )
        }
    }
    struct ParentView {
        child: Entity<ChildView>,
        parent_draws: usize,
    }
    impl Render for ParentView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            self.parent_draws += 1;
            div().size_full().child(self.child.clone())
        }
    }

    let paints = Rc::new(Cell::new(0usize));
    let paints_ref = paints.clone();
    let handle = cx.add_window(|_, cx| {
        let child = cx.new(|_| ChildView { paints: paints_ref });
        ParentView {
            child,
            parent_draws: 0,
        }
    });

    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    handle
        .update(cx, |_, window, _| {
            assert_eq!(window.rendered_frame.scene.layers.frames.len(), 1);
        })
        .unwrap();

    // Redraw parent while child stays retained and is replayed from previous scene.
    handle.update(cx, |_, _, cx| cx.notify()).unwrap();
    draw_window(cx, handle);
    assert_eq!(paints.get(), 1);
    handle
        .update(cx, |view, window, _| {
            assert_eq!(view.parent_draws, 2);
            assert_eq!(window.rendered_frame.scene.layers.frames.len(), 1);
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_alloc_key_skips_live_keys_on_wraparound(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct WrapView {
        count: usize,
    }
    impl Render for WrapView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let count = self.count;
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(10.), px(10.)), size(px(80.), px(80.)));
                        for id in 1..=count as u64 {
                            window.paint_cached_layer(
                                CachedLayerId::new(id),
                                bounds,
                                1,
                                CachedLayerOptions::default(),
                                |window| {
                                    window.paint_quad(fill(bounds, Hsla::red()));
                                },
                            );
                        }
                    },
                )
                .size_full(),
            )
        }
    }

    let handle = cx.add_window(|_, _| WrapView { count: 2 });
    draw_window(cx, handle);

    let (key1, key2) = handle
        .update(cx, |view, window, cx| {
            let k1 = window.fast_layers.cached.entries[&CachedLayerId::new(1)].key;
            let k2 = window.fast_layers.cached.entries[&CachedLayerId::new(2)].key;
            window.fast_layers.next_key = crate::fast::layers::scene::LAYER_KEY_LIMIT - 1;
            view.count = 3;
            cx.notify();
            (k1, k2)
        })
        .unwrap();
    draw_window(cx, handle);

    let key3 = handle
        .update(cx, |view, window, cx| {
            let k3 = window.fast_layers.cached.entries[&CachedLayerId::new(3)].key;
            assert_eq!(
                k3.0,
                crate::fast::layers::scene::LAYER_KEY_LIMIT - 1,
                "expected last key before wraparound"
            );
            view.count = 4;
            cx.notify();
            k3
        })
        .unwrap();
    draw_window(cx, handle);

    handle
        .update(cx, |_, window, _| {
            let key4 = window.fast_layers.cached.entries[&CachedLayerId::new(4)].key;
            assert!(key4.0 < crate::fast::layers::scene::LAYER_KEY_LIMIT);
            assert_ne!(key4, key1);
            assert_ne!(key4, key2);
            assert_ne!(key4, key3);
        })
        .unwrap();
}

#[crate::test]
fn cached_layer_clipped_away_expires_after_drop_frames(cx: &mut crate::TestAppContext) {
    if !crate::fast::layers::COMPILED {
        return;
    }
    struct ClippedView {
        clipped_away: bool,
    }
    impl Render for ClippedView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let clipped_away = self.clipped_away;
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, (), window, _| {
                        let bounds = Bounds::new(point(px(10.), px(10.)), size(px(80.), px(80.)));
                        let mask = if clipped_away {
                            Some(ContentMask {
                                bounds: Bounds::new(
                                    point(px(500.), px(500.)),
                                    size(px(40.), px(40.)),
                                ),
                            })
                        } else {
                            None
                        };
                        window.with_content_mask(mask, |window| {
                            window.paint_cached_layer(
                                CachedLayerId::new(30),
                                bounds,
                                1,
                                CachedLayerOptions::default(),
                                |window| {
                                    window.paint_quad(fill(bounds, Hsla::red()));
                                },
                            );
                        });
                    },
                )
                .size_full(),
            )
        }
    }

    let handle = cx.add_window(|_, _| ClippedView {
        clipped_away: false,
    });
    draw_window(cx, handle);
    handle
        .update(cx, |view, window, cx| {
            assert!(
                window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(30))
            );
            view.clipped_away = true;
            cx.notify();
        })
        .unwrap();

    for _ in 0..=crate::fast::layers::policy::DROP_AFTER_FRAMES {
        handle.update(cx, |_, _, cx| cx.notify()).unwrap();
        draw_window(cx, handle);
    }

    handle
        .update(cx, |_, window, _| {
            assert!(
                !window
                    .fast_layers
                    .cached
                    .entries
                    .contains_key(&CachedLayerId::new(30))
            );
        })
        .unwrap();
}
