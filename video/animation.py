from manim import *
import textwrap


class TrueSightPitch(Scene):

    # ============================================================
    # THEME
    # ============================================================

    BG = "#0e1117"
    PANEL = "#161b22"

    TEAL_C = "#2dd4bf"
    BLUE_C = "#60a5fa"
    GREEN_C = "#4ade80"
    YELLOW_C = "#facc15"
    RED_C = "#fb7185"
    PURPLE_C = "#c084fc"
    MUTED = "#64748b"

    # ============================================================
    # MASTER LAYOUT
    # ============================================================

    FACECAM_X = -4.85
    CONTENT_X = 2.45

    TITLE_Y = 2.70
    SUBTITLE_Y = 2.15
    DIAGRAM_Y = 0.35
    CAPTION_Y = -2.15

    def construct(self):

        self.camera.background_color = self.BG

        # ========================================================
        # FACECAM — PERSISTENT
        # ========================================================

        self.facecam_box = RoundedRectangle(
            width=4.15,
            height=6.75,
            corner_radius=0.15,
            stroke_color=self.TEAL_C,
            stroke_width=2,
        )

        self.facecam_box.move_to([
            self.FACECAM_X,
            0,
            0,
        ])

        self.facecam_label = Text(
            "FACECAM",
            font_size=17,
            color=self.MUTED,
        )

        self.facecam_label.move_to(self.facecam_box)

        self.play(
            Create(self.facecam_box),
            FadeIn(self.facecam_label),
            run_time=0.7,
        )

        # ========================================================
        # SCENE 1 — AI OR REAL?
        # ========================================================

        question = Text(
            "Can you tell which one is AI?",
            font_size=34,
            weight=BOLD,
        )

        question.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        self.play(
            Write(question),
            run_time=0.9,
        )

        # --------------------------------------------------------
        # LOAD ACTUAL IMAGES
        # --------------------------------------------------------

        image1 = ImageMobject("ai_cat.jpg")
        image2 = ImageMobject("real_cat.jpg")

        # Both images get the same maximum dimensions.
        # preserve_aspect_ratio=True prevents distortion.
        image1.set(
            height=2.65,
            width=2.65,
            preserve_aspect_ratio=True,
        )

        image2.set(
            height=2.65,
            width=2.65,
            preserve_aspect_ratio=True,
        )

        # --------------------------------------------------------
        # IDENTICAL IMAGE FRAMES
        # --------------------------------------------------------

        frame_width = 2.75
        frame_height = 2.85

        frame1 = RoundedRectangle(
            width=frame_width,
            height=frame_height,
            corner_radius=0.10,
            stroke_color=self.TEAL_C,
            stroke_width=2,
        )

        frame2 = RoundedRectangle(
            width=frame_width,
            height=frame_height,
            corner_radius=0.10,
            stroke_color=self.TEAL_C,
            stroke_width=2,
        )

        # Position image inside its frame
        image1.move_to(frame1.get_center())
        image2.move_to(frame2.get_center())

        # --------------------------------------------------------
        # NEUTRAL LABELS
        #
        # Don't reveal which one is AI.
        # --------------------------------------------------------

        label1 = Text(
            "IMAGE 1",
            font_size=17,
            color=WHITE,
            weight=BOLD,
        )

        label2 = Text(
            "IMAGE 2",
            font_size=17,
            color=WHITE,
            weight=BOLD,
        )

        label1.next_to(
            frame1,
            DOWN,
            buff=0.16,
        )

        label2.next_to(
            frame2,
            DOWN,
            buff=0.16,
        )

        # ImageMobject requires Group rather than VGroup
        card1 = Group(
            frame1,
            image1,
            label1,
        )

        card2 = Group(
            frame2,
            image2,
            label2,
        )

        # --------------------------------------------------------
        # ARRANGE CARDS
        # --------------------------------------------------------

        images = Group(
            card1,
            card2,
        )

        images.arrange(
            RIGHT,
            buff=0.55,
        )

        images.move_to([
            self.CONTENT_X,
            0.15,
            0,
        ])

        # --------------------------------------------------------
        # ANIMATE
        # --------------------------------------------------------

        self.play(
            FadeIn(frame1),
            FadeIn(image1),
            FadeIn(frame2),
            FadeIn(image2),
            run_time=0.9,
        )

        self.play(
            Write(label1),
            Write(label2),
            run_time=0.6,
        )

        self.wait(0.7)

        bet = Text(
            "I bet you couldn't.",
            font_size=26,
            color=self.TEAL_C,
        )

        bet.move_to([
            self.CONTENT_X,
            -2.45,
            0,
        ])

        self.play(
            Write(bet),
            run_time=0.8,
        )

        self.wait(1.3)
        # ========================================================
        # SCENE 2 — TRUESIGHT
        # ========================================================

        self.play(
            FadeOut(question),
            FadeOut(bet),
            images.animate
            .scale(0.52)
            .move_to([
                self.CONTENT_X,
                1.65,
                0,
            ]),
            run_time=0.9,
        )

        title = Text(
            "TrueSight",
            font_size=52,
            weight=BOLD,
            color=self.TEAL_C,
        )

        subtitle = Text(
            "AI Detection + Media Provenance",
            font_size=23,
        )

        brand = VGroup(
            title,
            subtitle,
        ).arrange(
            DOWN,
            buff=0.18,
        )

        brand.move_to([
            self.CONTENT_X,
            -0.25,
            0,
        ])

        self.play(Write(title))
        self.play(FadeIn(subtitle))

        tagline = Text(
            "Robust  •  Explainable  •  Compute-efficient",
            font_size=18,
            color=self.MUTED,
        )

        tagline.next_to(
            brand,
            DOWN,
            buff=0.5,
        )

        self.play(Write(tagline))

        self.wait(1.4)

        self.play(
            FadeOut(images),
            FadeOut(brand),
            FadeOut(tagline),
        )

        # ========================================================
        # SCENE 3 — MASTER PIPELINE
        # ========================================================

        pipeline_title = Text(
            "A 3-Tier Detection Pipeline",
            font_size=32,
            weight=BOLD,
        )

        pipeline_title.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        self.play(Write(pipeline_title))

        input_node = self.node(
            "INPUT IMAGE",
            1.6,
            0.62,
            WHITE,
            16,
        )

        tier1 = self.node(
            "TIER 1  •  C2PA",
            2.0,
            0.72,
            self.GREEN_C,
            16,
        )

        tier2 = self.node(
            "TIER 2  •  FORENSICS + VLM",
            2.8,
            0.72,
            self.YELLOW_C,
            15,
        )

        tier3 = self.node(
            "TIER 3  •  ConvNeXt",
            2.25,
            0.72,
            self.PURPLE_C,
            16,
        )

        fusion = self.node(
            "LATE FUSION",
            1.75,
            0.65,
            self.TEAL_C,
            16,
        )

        nodes = VGroup(
            input_node,
            tier1,
            tier2,
            tier3,
            fusion,
        )

        nodes.arrange(
            DOWN,
            buff=0.30,
        )

        nodes.move_to([
            self.CONTENT_X,
            -0.05,
            0,
        ])

        pipeline_arrows = VGroup()

        for first, second in zip(
            nodes[:-1],
            nodes[1:],
        ):

            arrow = Arrow(
                first.get_bottom(),
                second.get_top(),
                buff=0.06,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.12,
            )

            pipeline_arrows.add(arrow)

        self.play(FadeIn(input_node))

        for i in range(4):

            self.play(
                GrowArrow(pipeline_arrows[i]),
                FadeIn(nodes[i + 1]),
                run_time=0.45,
            )

        verdict = Text(
            "VERDICT",
            font_size=18,
            color=self.TEAL_C,
            weight=BOLD,
        )

        verdict.next_to(
            fusion,
            DOWN,
            buff=0.38,
        )

        verdict_arrow = Arrow(
            fusion.get_bottom(),
            verdict.get_top(),
            buff=0.07,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.10,
        )

        self.play(
            GrowArrow(verdict_arrow),
            Write(verdict),
        )

        pipeline = VGroup(
            pipeline_title,
            nodes,
            pipeline_arrows,
            verdict,
            verdict_arrow,
        )

        self.wait(1.2)

        # ========================================================
        # ENTER FOCUS MODE
        #
        # Strong dimming.
        # No blur.
        # ========================================================

        self.dim_background(opacity=0.94)

        # ========================================================
        # SCENE 4 — C2PA
        # ========================================================

        scene4_title = self.scene_title(
            "Cryptographic Provenance",
            "TIER 1  •  C2PA",
            self.GREEN_C,
        )

        self.play(
            Write(scene4_title[0]),
            FadeIn(scene4_title[1]),
        )

        image_node = self.node(
            "IMAGE",
            1.25,
            0.85,
            self.BLUE_C,
            16,
        )

        credentials = self.node(
            "C2PA\nCREDENTIALS",
            1.55,
            1.0,
            self.GREEN_C,
            15,
        )

        verify = self.node(
            "VERIFY\nSIGNATURE",
            1.55,
            1.0,
            self.TEAL_C,
            15,
        )

        c2pa_nodes = VGroup(
            image_node,
            credentials,
            verify,
        ).arrange(
            RIGHT,
            buff=0.70,
        )

        c2pa_nodes.move_to([
            self.CONTENT_X,
            self.DIAGRAM_Y,
            0,
        ])

        arrow1 = self.horizontal_arrow(
            image_node,
            credentials,
        )

        arrow2 = self.horizontal_arrow(
            credentials,
            verify,
        )

        self.play(FadeIn(image_node))

        self.play(
            GrowArrow(arrow1),
            FadeIn(credentials),
        )

        self.play(
            GrowArrow(arrow2),
            FadeIn(verify),
        )

        deterministic = Text(
            "DETERMINISTIC PATH",
            font_size=20,
            color=self.GREEN_C,
            weight=BOLD,
        )

        deterministic.move_to([
            self.CONTENT_X,
            -1.20,
            0,
        ])

        benefits = Text(
            "Fast   •   Verifiable   •   Low Compute",
            font_size=17,
            color=WHITE,
        )

        benefits.next_to(
            deterministic,
            DOWN,
            buff=0.20,
        )

        self.play(Write(deterministic))
        self.play(Write(benefits))

        route = Text(
            "No credentials?  →  Route to Tier 2",
            font_size=18,
            color=self.YELLOW_C,
        )

        route.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(Write(route))

        self.wait(1)

        scene4 = VGroup(
            scene4_title,
            c2pa_nodes,
            arrow1,
            arrow2,
            deterministic,
            benefits,
            route,
        )

        self.play(FadeOut(scene4))

        # ========================================================
        # SCENE 5 — BLIND FORENSICS
        # ========================================================

        scene5_title = self.scene_title(
            "Blind Forensics",
            "TIER 2  •  DETERMINISTIC SIGNALS",
            self.YELLOW_C,
        )

        self.play(
            Write(scene5_title[0]),
            FadeIn(scene5_title[1]),
        )

        source = self.node(
            "IMAGE",
            1.15,
            0.82,
            self.BLUE_C,
            16,
        )

        source.move_to([
            self.CONTENT_X - 2.45,
            self.DIAGRAM_Y,
            0,
        ])

        ela = self.node(
            "ELA",
            1.35,
            0.60,
            self.YELLOW_C,
            15,
        )

        high_pass = self.node(
            "HIGH-PASS",
            1.35,
            0.60,
            self.YELLOW_C,
            14,
        )

        orb = self.node(
            "ORB",
            1.35,
            0.60,
            self.YELLOW_C,
            15,
        )

        methods = VGroup(
            ela,
            high_pass,
            orb,
        ).arrange(
            DOWN,
            buff=0.20,
        )

        methods.move_to([
            self.CONTENT_X,
            self.DIAGRAM_Y,
            0,
        ])

        mask = self.create_mask()

        mask.move_to([
            self.CONTENT_X + 2.45,
            self.DIAGRAM_Y,
            0,
        ])

        input_arrows = VGroup()

        for method in methods:

            arrow = Arrow(
                source.get_right(),
                method.get_left(),
                buff=0.08,
                stroke_width=1.8,
                max_tip_length_to_length_ratio=0.08,
            )

            input_arrows.add(arrow)

        output_arrows = VGroup()

        for method in methods:

            arrow = Arrow(
                method.get_right(),
                mask.get_left(),
                buff=0.08,
                stroke_width=1.8,
                max_tip_length_to_length_ratio=0.08,
            )

            output_arrows.add(arrow)

        self.play(FadeIn(source))

        self.play(
            *[GrowArrow(a) for a in input_arrows],
            *[FadeIn(m) for m in methods],
            run_time=0.8,
        )

        self.play(
            *[GrowArrow(a) for a in output_arrows],
            FadeIn(mask),
            run_time=0.8,
        )

        mask_label = Text(
            "TAMPER MASK",
            font_size=17,
            color=self.RED_C,
            weight=BOLD,
        )

        mask_label.next_to(
            mask,
            DOWN,
            buff=0.18,
        )

        self.play(Write(mask_label))

        fusion_text = Text(
            "Deterministic Signal Fusion",
            font_size=20,
            color=self.YELLOW_C,
        )

        fusion_text.move_to([
            self.CONTENT_X,
            -1.35,
            0,
        ])

        question_text = Text(
            "Where does the image look manipulated?",
            font_size=17,
            color=WHITE,
        )

        question_text.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(Write(fusion_text))
        self.play(Write(question_text))

        self.wait(1)

        scene5 = VGroup(
            scene5_title,
            source,
            methods,
            mask,
            input_arrows,
            output_arrows,
            mask_label,
            fusion_text,
            question_text,
        )

        self.play(FadeOut(scene5))

        # ========================================================
        # SCENE 6 — VLM
        # ========================================================

        scene6_title = self.scene_title(
            "Semantic Analysis",
            "TIER 2  •  VISION LANGUAGE MODEL",
            self.TEAL_C,
        )

        self.play(
            Write(scene6_title[0]),
            FadeIn(scene6_title[1]),
        )

        mask2 = self.create_mask()

        vlm = self.node(
            "VISION\nLANGUAGE\nMODEL",
            1.75,
            1.25,
            self.TEAL_C,
            15,
        )

        semantic = self.node(
            "SEMANTIC\nREVIEW",
            1.65,
            1.05,
            self.GREEN_C,
            15,
        )

        vlm_flow = VGroup(
            mask2,
            vlm,
            semantic,
        ).arrange(
            RIGHT,
            buff=0.75,
        )

        vlm_flow.move_to([
            self.CONTENT_X,
            0.55,
            0,
        ])

        vlm_arrow1 = self.horizontal_arrow(
            mask2,
            vlm,
        )

        vlm_arrow2 = self.horizontal_arrow(
            vlm,
            semantic,
        )

        self.play(FadeIn(mask2))

        self.play(
            GrowArrow(vlm_arrow1),
            FadeIn(vlm),
        )

        self.play(
            GrowArrow(vlm_arrow2),
            FadeIn(semantic),
        )

        semantic_items = VGroup(
            Text(
                "Impossible lighting",
                font_size=17,
            ),
            Text(
                "Anatomical inconsistencies",
                font_size=17,
            ),
            Text(
                "Texture anomalies",
                font_size=17,
            ),
        ).arrange(
            DOWN,
            buff=0.15,
        )

        semantic_items.move_to([
            self.CONTENT_X,
            -1.20,
            0,
        ])

        for item in semantic_items:

            self.play(
                Write(item),
                run_time=0.35,
            )

        caption6 = Text(
            "The mask tells us where. The VLM tells us why.",
            font_size=17,
            color=self.TEAL_C,
        )

        caption6.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(Write(caption6))

        self.wait(1)

        scene6 = VGroup(
            scene6_title,
            vlm_flow,
            vlm_arrow1,
            vlm_arrow2,
            semantic_items,
            caption6,
        )

        self.play(FadeOut(scene6))

        # ========================================================
        # SCENE 7 — CONVNEXT
        # ========================================================

        scene7_title = self.scene_title(
            "Deepfake Detection",
            "TIER 3  •  ConvNeXt",
            self.PURPLE_C,
        )

        self.play(
            Write(scene7_title[0]),
            FadeIn(scene7_title[1]),
        )

        conv_image = self.node(
            "IMAGE",
            1.05,
            0.82,
            self.BLUE_C,
            15,
        )

        feature_node = self.node(
            "FEATURE\nEXTRACTION",
            1.45,
            1.05,
            self.PURPLE_C,
            14,
        )

        convnext = self.node(
            "ConvNeXt",
            1.40,
            1.05,
            self.PURPLE_C,
            16,
        )

        score = self.node(
            "AI\n0.94",
            1.05,
            1.05,
            self.RED_C,
            16,
        )

        conv_nodes = VGroup(
            conv_image,
            feature_node,
            convnext,
            score,
        ).arrange(
            RIGHT,
            buff=0.45,
        )

        conv_nodes.move_to([
            self.CONTENT_X,
            self.DIAGRAM_Y,
            0,
        ])

        conv_arrows = VGroup()

        for first, second in zip(
            conv_nodes[:-1],
            conv_nodes[1:],
        ):

            conv_arrows.add(
                self.horizontal_arrow(
                    first,
                    second,
                )
            )

        self.play(FadeIn(conv_image))

        for i in range(3):

            self.play(
                GrowArrow(conv_arrows[i]),
                FadeIn(conv_nodes[i + 1]),
                run_time=0.45,
            )

        latent = Text(
            "Latent Frequency Artifacts",
            font_size=21,
            color=self.PURPLE_C,
            weight=BOLD,
        )

        latent.move_to([
            self.CONTENT_X,
            -1.25,
            0,
        ])

        generators = Text(
            "Diffusion  •  GANs  •  Synthetic Generators",
            font_size=16,
            color=WHITE,
        )

        generators.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(Write(latent))
        self.play(Write(generators))

        self.wait(1)

        scene7 = VGroup(
            scene7_title,
            conv_nodes,
            conv_arrows,
            latent,
            generators,
        )

        self.play(FadeOut(scene7))

        # ========================================================
        # SCENE 8 — GRAD-CAM
        # ========================================================

        scene8_title = self.scene_title(
            "Explainable AI",
            "GRAD-CAM  •  NOT JUST A BLACK BOX",
            self.RED_C,
        )

        self.play(
            Write(scene8_title[0]),
            FadeIn(scene8_title[1]),
        )

        grad_image = self.node(
            "IMAGE",
            1.20,
            0.90,
            self.BLUE_C,
            16,
        )

        grad_network = self.node(
            "ConvNeXt",
            1.55,
            1.05,
            self.PURPLE_C,
            16,
        )

        heatmap = self.create_heatmap()

        grad_nodes = VGroup(
            grad_image,
            grad_network,
            heatmap,
        ).arrange(
            RIGHT,
            buff=0.75,
        )

        grad_nodes.move_to([
            self.CONTENT_X,
            self.DIAGRAM_Y,
            0,
        ])

        grad_arrow1 = self.horizontal_arrow(
            grad_image,
            grad_network,
        )

        grad_arrow2 = self.horizontal_arrow(
            grad_network,
            heatmap,
        )

        self.play(FadeIn(grad_image))

        self.play(
            GrowArrow(grad_arrow1),
            FadeIn(grad_network),
        )

        self.play(
            GrowArrow(grad_arrow2),
            FadeIn(heatmap),
        )

        gradient_label = Text(
            "Gradient-weighted activation",
            font_size=18,
            color=self.RED_C,
        )

        gradient_label.move_to([
            self.CONTENT_X,
            -1.25,
            0,
        ])

        grad_caption = Text(
            "Which pixels drove the AI classification?",
            font_size=17,
            color=WHITE,
        )

        grad_caption.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(Write(gradient_label))
        self.play(Write(grad_caption))

        self.wait(1)

        scene8 = VGroup(
            scene8_title,
            grad_nodes,
            grad_arrow1,
            grad_arrow2,
            gradient_label,
            grad_caption,
        )

        self.play(FadeOut(scene8))

        # ========================================================
        # SCENE 9 — LATE FUSION
        # ========================================================

        scene9_title = self.scene_title(
            "Late Fusion",
            "CALIBRATED MULTI-SIGNAL DECISION",
            self.TEAL_C,
        )

        self.play(
            Write(scene9_title[0]),
            FadeIn(scene9_title[1]),
        )

        provenance = self.node(
            "PROVENANCE\nC2PA",
            1.55,
            0.95,
            self.GREEN_C,
            14,
        )

        forensic = self.node(
            "FORENSICS\n+ VLM",
            1.55,
            0.95,
            self.YELLOW_C,
            14,
        )

        conv_score = self.node(
            "ConvNeXt\n0.94",
            1.55,
            0.95,
            self.PURPLE_C,
            14,
        )

        scores = VGroup(
            provenance,
            forensic,
            conv_score,
        ).arrange(
            RIGHT,
            buff=0.45,
        )

        scores.move_to([
            self.CONTENT_X,
            0.95,
            0,
        ])

        self.play(
            FadeIn(provenance),
            FadeIn(forensic),
            FadeIn(conv_score),
        )

        # Move fusion slightly lower to give arrows more room
        fusion_node = self.node(
            "CALIBRATED\nFUSION",
            1.80,
            1.0,
            self.TEAL_C,
            15,
        )

        fusion_node.move_to([
            self.CONTENT_X,
            -0.70,
            0,
        ])

        # --------------------------------------------------------
        # FIXED FUSION ARROWS
        #
        # Each arrow gets its own landing point.
        # Smaller tips.
        # --------------------------------------------------------

        fusion_arrows = VGroup()

        target_points = [
            fusion_node.get_top() + LEFT * 0.52,
            fusion_node.get_top(),
            fusion_node.get_top() + RIGHT * 0.52,
        ]

        for score_node, target in zip(
            scores,
            target_points,
        ):

            arrow = Arrow(
                start=score_node.get_bottom(),
                end=target,
                buff=0.12,
                stroke_width=1.5,
                max_tip_length_to_length_ratio=0.07,
                max_stroke_width_to_length_ratio=4,
            )

            fusion_arrows.add(arrow)

        self.play(
            *[
                GrowArrow(a)
                for a in fusion_arrows
            ],
            FadeIn(fusion_node),
            run_time=0.9,
        )

        result = Text(
            "AI GENERATED  •  94% CONFIDENCE",
            font_size=21,
            color=self.RED_C,
            weight=BOLD,
        )

        result.move_to([
            self.CONTENT_X,
            -1.78,
            0,
        ])

        self.play(Write(result))

        fusion_caption = Text(
            "Evidence from multiple independent signals.",
            font_size=16,
            color=WHITE,
        )

        fusion_caption.move_to([
            self.CONTENT_X,
            -2.50,
            0,
        ])

        self.play(Write(fusion_caption))

        # --------------------------------------------------------
        # IMPORTANT:
        # Give late fusion enough time to be understood.
        # --------------------------------------------------------

        self.wait(2.5)

        scene9 = VGroup(
            scene9_title,
            scores,
            fusion_arrows,
            fusion_node,
            result,
            fusion_caption,
        )

        # ========================================================
        # SLOW TRANSITION OUT OF FOCUS MODE
        # ========================================================

        # First remove ONLY the foreground explanation.
        self.play(
            FadeOut(scene9),
            run_time=1.2,
        )

        # For a moment, only the heavily dimmed architecture
        # remains.
        self.wait(0.8)

        # Slowly reveal the complete pipeline again.
        self.play(
            FadeOut(self.dim_overlay),
            run_time=1.3,
        )

        # Let the viewer reconnect all three tiers.
        self.wait(2.0)

        # Now close the algorithm section.
        self.play(
            FadeOut(pipeline),
            run_time=1.2,
        )

        # Visual breathing room between algorithm and product.
        self.wait(0.8)

        # ========================================================
        # SCENE 10 — PRODUCT DEMO 1
        # ========================================================

        demo1 = self.demo_screen(
            "PRODUCT DEMO 1",
            "Upload + Provenance Check",
            "Insert demo_1.mp4 here in post",
        )

        demo1.move_to([
            self.CONTENT_X,
            0,
            0,
        ])

        self.play(
            FadeIn(
                demo1,
                shift=RIGHT * 0.15,
            ),
            run_time=1.2,
        )

        self.wait(3)

        # ========================================================
        # SCENE 11 — PRODUCT DEMO 2
        # ========================================================

        demo2 = self.demo_screen(
            "PRODUCT DEMO 2",
            "Forensics + AI Analysis",
            "Insert demo_2.mp4 here in post",
        )

        demo2.move_to([
            self.CONTENT_X,
            0,
            0,
        ])

        self.play(
            ReplacementTransform(
                demo1,
                demo2,
            ),
            run_time=1.0,
        )

        self.wait(3)

        self.play(
            FadeOut(demo2),
            run_time=0.8,
        )

        # ========================================================
        # SCENE 12 — RESULT DASHBOARD
        # ========================================================

        result_title = Text(
            "TrueSight Result",
            font_size=34,
            color=self.TEAL_C,
            weight=BOLD,
        )

        result_title.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        self.play(Write(result_title))

        verdict_box = RoundedRectangle(
            width=6.75,
            height=0.90,
            corner_radius=0.10,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        verdict_text = Text(
            "AI GENERATED   •   94% CONFIDENCE",
            font_size=21,
            color=self.RED_C,
            weight=BOLD,
        )

        verdict_text.move_to(
            verdict_box
        )

        verdict_group = VGroup(
            verdict_box,
            verdict_text,
        )

        verdict_group.move_to([
            self.CONTENT_X,
            1.65,
            0,
        ])

        self.play(Create(verdict_box))
        self.play(Write(verdict_text))

        output_data = [
            (
                "SOURCE",
                "Synthetic",
            ),
            (
                "PROVENANCE",
                "No verified C2PA",
            ),
            (
                "FORENSICS",
                "Manipulation detected",
            ),
            (
                "VLM REVIEW",
                "Semantic anomalies",
            ),
            (
                "ConvNeXt",
                "0.94 AI probability",
            ),
            (
                "EXPLAINABILITY",
                "Grad-CAM available",
            ),
        ]

        cards = VGroup()

        for heading, value in output_data:

            card = self.result_card(
                heading,
                value,
            )

            cards.add(card)

        cards.arrange_in_grid(
            rows=3,
            cols=2,
            buff=(0.28, 0.25),
        )

        cards.move_to([
            self.CONTENT_X,
            -0.35,
            0,
        ])

        for card in cards:

            self.play(
                FadeIn(
                    card,
                    shift=UP * 0.08,
                ),
                run_time=0.25,
            )

        self.wait(2)

        # ========================================================
        # SCENE 13 — END
        # ========================================================

        self.play(
            FadeOut(cards),
            FadeOut(verdict_group),
            FadeOut(result_title),
            run_time=0.8,
        )

        final_title = Text(
            "TrueSight",
            font_size=54,
            color=self.TEAL_C,
            weight=BOLD,
        )

        final_subtitle = Text(
            "Know what's real.",
            font_size=27,
        )

        final_group = VGroup(
            final_title,
            final_subtitle,
        ).arrange(
            DOWN,
            buff=0.25,
        )

        final_group.move_to([
            self.CONTENT_X,
            0,
            0,
        ])

        self.play(
            Write(final_title),
            run_time=1.0,
        )

        self.play(
            Write(final_subtitle),
            run_time=0.8,
        )

        self.wait(2)

        self.play(
            FadeOut(final_group),
            FadeOut(self.facecam_box),
            FadeOut(self.facecam_label),
            run_time=1.0,
        )

    # ============================================================
    # HELPER — CONSISTENT SCENE TITLE
    # ============================================================

    def scene_title(
        self,
        title,
        subtitle,
        accent,
    ):

        title_obj = Text(
            title,
            font_size=31,
            weight=BOLD,
        )

        subtitle_obj = Text(
            subtitle,
            font_size=15,
            color=accent,
            weight=BOLD,
        )

        title_obj.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        subtitle_obj.move_to([
            self.CONTENT_X,
            self.SUBTITLE_Y,
            0,
        ])

        return VGroup(
            title_obj,
            subtitle_obj,
        )

    # ============================================================
    # HELPER — STRONG DIM
    # ============================================================

    def dim_background(
        self,
        opacity=0.94,
    ):

        # No blur.
        # Covers right-side presentation area only.
        # Facecam remains unaffected.

        self.dim_overlay = Rectangle(
            width=9.25,
            height=config.frame_height,
            stroke_width=0,
            fill_color=self.BG,
            fill_opacity=opacity,
        )

        self.dim_overlay.to_edge(
            RIGHT,
            buff=0,
        )

        self.play(
            FadeIn(self.dim_overlay),
            run_time=0.35,
        )

    # ============================================================
    # HELPER — NODE
    # ============================================================

    def node(
        self,
        text,
        width,
        height,
        color,
        font_size=16,
    ):

        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.09,
            stroke_color=color,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.96,
        )

        label = Text(
            text,
            font_size=font_size,
            color=color,
            line_spacing=0.85,
        )

        # Prevent text overflow
        if label.width > width * 0.82:

            label.scale_to_fit_width(
                width * 0.82
            )

        if label.height > height * 0.72:

            label.scale_to_fit_height(
                height * 0.72
            )

        label.move_to(box)

        return VGroup(
            box,
            label,
        )

    # ============================================================
    # HELPER — HORIZONTAL ARROW
    # ============================================================

    def horizontal_arrow(
        self,
        first,
        second,
    ):

        return Arrow(
            first.get_right(),
            second.get_left(),
            buff=0.08,
            stroke_width=1.8,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=4,
        )

    # ============================================================
    # HELPER — OPENING IMAGE CARD
    # ============================================================

    def image_card(
        self,
        title,
        subtitle,
        color,
    ):

        box = RoundedRectangle(
            width=2.35,
            height=2.70,
            corner_radius=0.12,
            stroke_color=color,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=1,
        )

        head = Circle(
            radius=0.44,
            color=color,
        )

        head.move_to(
            box.get_center()
            + UP * 0.35
        )

        shoulders = Arc(
            radius=0.68,
            start_angle=0,
            angle=PI,
            color=color,
        )

        shoulders.rotate(PI)

        shoulders.move_to(
            box.get_center()
            + DOWN * 0.55
        )

        title_text = Text(
            title,
            font_size=16,
        )

        title_text.next_to(
            box,
            UP,
            buff=0.13,
        )

        subtitle_text = Text(
            subtitle,
            font_size=17,
            color=color,
            weight=BOLD,
        )

        subtitle_text.next_to(
            box,
            DOWN,
            buff=0.13,
        )

        return VGroup(
            box,
            head,
            shoulders,
            title_text,
            subtitle_text,
        )

    # ============================================================
    # HELPER — FORENSIC MASK
    # ============================================================

    def create_mask(self):

        box = Square(
            side_length=1.35,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        pixels = VGroup()

        positions = [
            (-0.34, 0.28),
            (0.02, 0.36),
            (0.34, 0.12),
            (-0.20, -0.05),
            (0.14, -0.25),
            (0.37, -0.34),
        ]

        for x, y in positions:

            pixel = Square(
                side_length=0.20,
                stroke_width=0,
                fill_color=self.RED_C,
                fill_opacity=0.72,
            )

            pixel.move_to(
                box.get_center()
                + RIGHT * x
                + UP * y
            )

            pixels.add(pixel)

        return VGroup(
            box,
            pixels,
        )

    # ============================================================
    # HELPER — GRAD-CAM
    # ============================================================

    def create_heatmap(self):

        box = Square(
            side_length=1.35,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        heat = VGroup()

        for radius, opacity in [
            (0.50, 0.12),
            (0.34, 0.27),
            (0.19, 0.55),
        ]:

            circle = Circle(
                radius=radius,
                stroke_width=0,
                fill_color=self.RED_C,
                fill_opacity=opacity,
            )

            circle.move_to(
                box.get_center()
                + RIGHT * 0.12
                + UP * 0.08
            )

            heat.add(circle)

        label = Text(
            "GRAD-CAM",
            font_size=13,
            color=self.RED_C,
            weight=BOLD,
        )

        label.next_to(
            box,
            DOWN,
            buff=0.13,
        )

        return VGroup(
            box,
            heat,
            label,
        )

    # ============================================================
    # HELPER — PRODUCT DEMO WINDOW
    # ============================================================

    def demo_screen(
        self,
        title,
        subtitle,
        instruction,
    ):

        outer = RoundedRectangle(
            width=5.75,
            height=4.95,
            corner_radius=0.15,
            stroke_color=self.TEAL_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.25,
        )

        top_bar = Rectangle(
            width=5.75,
            height=0.48,
            stroke_width=0,
            fill_color=self.PANEL,
            fill_opacity=1,
        )

        top_bar.move_to(
            outer.get_top()
            + DOWN * 0.24
        )

        dots = VGroup(
            Dot(radius=0.05),
            Dot(radius=0.05),
            Dot(radius=0.05),
        ).arrange(
            RIGHT,
            buff=0.09,
        )

        dots.move_to(
            top_bar.get_left()
            + RIGHT * 0.40
        )

        heading = Text(
            title,
            font_size=23,
            color=self.TEAL_C,
            weight=BOLD,
        )

        heading.move_to(
            outer.get_center()
            + UP * 1.40
        )

        subtitle_text = Text(
            subtitle,
            font_size=18,
        )

        subtitle_text.next_to(
            heading,
            DOWN,
            buff=0.20,
        )

        video = RoundedRectangle(
            width=4.55,
            height=2.0,
            corner_radius=0.08,
            stroke_color=self.MUTED,
            stroke_width=1.5,
        )

        video.move_to(
            outer.get_center()
            + DOWN * 0.55
        )

        play = Triangle(
            color=self.TEAL_C,
            fill_color=self.TEAL_C,
            fill_opacity=1,
        )

        play.scale(0.19)
        play.rotate(-PI / 2)
        play.move_to(video)

        note = Text(
            instruction,
            font_size=12,
            color=self.MUTED,
        )

        note.next_to(
            video,
            DOWN,
            buff=0.17,
        )

        return VGroup(
            outer,
            top_bar,
            dots,
            heading,
            subtitle_text,
            video,
            play,
            note,
        )

    # ============================================================
    # HELPER — RESULT CARD
    # ============================================================

    def result_card(
        self,
        heading,
        value,
    ):

        box = RoundedRectangle(
            width=3.2,
            height=0.72,
            corner_radius=0.07,
            stroke_color=self.MUTED,
            stroke_width=1.2,
            fill_color=self.PANEL,
            fill_opacity=0.9,
        )

        heading_text = Text(
            heading,
            font_size=11,
            color=self.TEAL_C,
            weight=BOLD,
        )

        value_text = Text(
            value,
            font_size=12,
        )

        if value_text.width > 2.15:

            value_text.scale_to_fit_width(
                2.15
            )

        text = VGroup(
            heading_text,
            value_text,
        ).arrange(
            DOWN,
            buff=0.07,
        )

        text.move_to(box)

        return VGroup(
            box,
            text,
        )