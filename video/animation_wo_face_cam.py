from manim import *


class TrueSightPitch(Scene):

    # ============================================================
    # FILES
    # Change these to your actual uploaded filenames
    # ============================================================

    REAL_IMAGE = "real_cat.jpg"
    AI_IMAGE = "ai_cat.jpg"

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
    # Everything is now centered on the full screen.
    # ============================================================

    CONTENT_X = 0

    TITLE_Y = 2.75
    SUBTITLE_Y = 2.15
    DIAGRAM_Y = 0.35
    CAPTION_Y = -2.25

    def construct(self):

        self.camera.background_color = self.BG

        # ========================================================
        # SCENE 1 — AI OR REAL?
        # ========================================================

        question = Text(
            "Can you tell which one is AI?",
            font_size=38,
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
        # ACTUAL IMAGES
        # --------------------------------------------------------

        image1 = ImageMobject(self.REAL_IMAGE)
        image2 = ImageMobject(self.AI_IMAGE)

        # Keep aspect ratios intact.
        image1.scale_to_fit_height(3.0)
        image2.scale_to_fit_height(3.0)

        # Prevent very wide images from becoming oversized.
        if image1.width > 4.2:
            image1.scale_to_fit_width(4.2)

        if image2.width > 4.2:
            image2.scale_to_fit_width(4.2)

        # Borders around the actual image dimensions.
        frame1 = SurroundingRectangle(
            image1,
            color=self.TEAL_C,
            buff=0.05,
            stroke_width=2,
        )

        frame2 = SurroundingRectangle(
            image2,
            color=self.TEAL_C,
            buff=0.05,
            stroke_width=2,
        )

        card1 = Group(
            image1,
            frame1,
        )

        card2 = Group(
            image2,
            frame2,
        )

        images = Group(
            card1,
            card2,
        )

        images.arrange(
            RIGHT,
            buff=0.9,
        )

        images.move_to([
            self.CONTENT_X,
            0.15,
            0,
        ])

        label1 = Text(
            "IMAGE 1",
            font_size=18,
            weight=BOLD,
        )

        label2 = Text(
            "IMAGE 2",
            font_size=18,
            weight=BOLD,
        )

        label1.next_to(
            card1,
            DOWN,
            buff=0.15,
        )

        label2.next_to(
            card2,
            DOWN,
            buff=0.15,
        )

        self.play(
            FadeIn(card1, shift=UP * 0.15),
            FadeIn(card2, shift=UP * 0.15),
            run_time=0.9,
        )

        self.play(
            Write(label1),
            Write(label2),
            run_time=0.5,
        )

        bet = Text(
            "I bet you couldn't.",
            font_size=28,
            color=self.TEAL_C,
        )

        bet.move_to([
            self.CONTENT_X,
            -2.75,
            0,
        ])

        self.play(
            Write(bet),
            run_time=0.7,
        )

        self.wait(1.2)

        # Add labels to image group so they transition together.
        images.add(
            label1,
            label2,
        )

        # ========================================================
        # SCENE 2 — TRUESIGHT
        # ========================================================

        self.play(
            FadeOut(question),
            FadeOut(bet),

            images.animate
            .scale(0.48)
            .move_to([
                self.CONTENT_X,
                1.55,
                0,
            ]),

            run_time=0.9,
        )

        title = Text(
            "TrueSight",
            font_size=58,
            weight=BOLD,
            color=self.TEAL_C,
        )

        subtitle = Text(
            "AI Detection + Media Provenance",
            font_size=25,
        )

        brand = VGroup(
            title,
            subtitle,
        )

        brand.arrange(
            DOWN,
            buff=0.18,
        )

        brand.move_to([
            self.CONTENT_X,
            -0.35,
            0,
        ])

        self.play(
            Write(title),
            run_time=0.8,
        )

        self.play(
            FadeIn(subtitle),
            run_time=0.5,
        )

        tagline = Text(
            "Robust  •  Explainable  •  Compute-efficient",
            font_size=19,
            color=self.MUTED,
        )

        tagline.next_to(
            brand,
            DOWN,
            buff=0.45,
        )

        self.play(
            Write(tagline),
            run_time=0.7,
        )

        self.wait(1)

        self.play(
            FadeOut(images),
            FadeOut(brand),
            FadeOut(tagline),
            run_time=0.7,
        )

        # ========================================================
        # SCENE 3 — MASTER PIPELINE
        # ========================================================

        pipeline_title = Text(
            "A 3-Tier Detection Pipeline",
            font_size=36,
            weight=BOLD,
        )

        pipeline_title.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        self.play(
            Write(pipeline_title),
            run_time=0.8,
        )

        input_node = self.node(
            "INPUT IMAGE",
            2.0,
            0.65,
            WHITE,
            17,
        )

        tier1 = self.node(
            "TIER 1  •  C2PA",
            2.4,
            0.72,
            self.GREEN_C,
            17,
        )

        tier2 = self.node(
            "TIER 2  •  FORENSICS + VLM",
            3.3,
            0.72,
            self.YELLOW_C,
            16,
        )

        tier3 = self.node(
            "TIER 3  •  ConvNeXt",
            2.7,
            0.72,
            self.PURPLE_C,
            17,
        )

        fusion = self.node(
            "LATE FUSION",
            2.0,
            0.65,
            self.TEAL_C,
            17,
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
            buff=0.28,
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
                max_tip_length_to_length_ratio=0.10,
            )

            pipeline_arrows.add(arrow)

        self.play(
            FadeIn(input_node),
            run_time=0.4,
        )

        for i in range(4):

            self.play(
                GrowArrow(pipeline_arrows[i]),
                FadeIn(nodes[i + 1]),
                run_time=0.4,
            )

        verdict = Text(
            "VERDICT",
            font_size=19,
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
            max_tip_length_to_length_ratio=0.08,
        )

        self.play(
            GrowArrow(verdict_arrow),
            Write(verdict),
            run_time=0.5,
        )

        pipeline = VGroup(
            pipeline_title,
            nodes,
            pipeline_arrows,
            verdict,
            verdict_arrow,
        )

        self.wait(1.3)

        # ========================================================
        # ENTER FOCUS MODE
        #
        # No blur.
        # Strong 94% dim over ENTIRE screen.
        # ========================================================

        self.dim_background(
            opacity=0.94
        )

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
            run_time=0.7,
        )

        image_node = self.node(
            "IMAGE",
            1.5,
            0.9,
            self.BLUE_C,
            17,
        )

        credentials = self.node(
            "C2PA\nCREDENTIALS",
            1.9,
            1.1,
            self.GREEN_C,
            16,
        )

        verify = self.node(
            "VERIFY\nSIGNATURE",
            1.9,
            1.1,
            self.TEAL_C,
            16,
        )

        c2pa_nodes = VGroup(
            image_node,
            credentials,
            verify,
        )

        c2pa_nodes.arrange(
            RIGHT,
            buff=1.0,
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

        self.play(
            FadeIn(image_node),
            run_time=0.4,
        )

        self.play(
            GrowArrow(arrow1),
            FadeIn(credentials),
            run_time=0.5,
        )

        self.play(
            GrowArrow(arrow2),
            FadeIn(verify),
            run_time=0.5,
        )

        deterministic = Text(
            "DETERMINISTIC PATH",
            font_size=22,
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
            font_size=18,
        )

        benefits.next_to(
            deterministic,
            DOWN,
            buff=0.20,
        )

        self.play(
            Write(deterministic),
            run_time=0.5,
        )

        self.play(
            Write(benefits),
            run_time=0.5,
        )

        route = Text(
            "No credentials?  →  Route to Tier 2",
            font_size=19,
            color=self.YELLOW_C,
        )

        route.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(
            Write(route),
            run_time=0.6,
        )

        self.wait(0.8)

        scene4 = VGroup(
            scene4_title,
            c2pa_nodes,
            arrow1,
            arrow2,
            deterministic,
            benefits,
            route,
        )

        self.play(
            FadeOut(scene4),
            run_time=0.6,
        )

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
            run_time=0.7,
        )

        source = self.node(
            "IMAGE",
            1.4,
            0.9,
            self.BLUE_C,
            17,
        )

        source.move_to([
            -4.2,
            self.DIAGRAM_Y,
            0,
        ])

        ela = self.node(
            "ELA",
            1.6,
            0.62,
            self.YELLOW_C,
            16,
        )

        high_pass = self.node(
            "HIGH-PASS",
            1.6,
            0.62,
            self.YELLOW_C,
            15,
        )

        orb = self.node(
            "ORB",
            1.6,
            0.62,
            self.YELLOW_C,
            16,
        )

        methods = VGroup(
            ela,
            high_pass,
            orb,
        )

        methods.arrange(
            DOWN,
            buff=0.22,
        )

        methods.move_to([
            0,
            self.DIAGRAM_Y,
            0,
        ])

        mask = self.create_mask()

        mask.move_to([
            4.2,
            self.DIAGRAM_Y,
            0,
        ])

        input_arrows = VGroup()

        for method in methods:

            input_arrows.add(
                Arrow(
                    source.get_right(),
                    method.get_left(),
                    buff=0.12,
                    stroke_width=1.7,
                    max_tip_length_to_length_ratio=0.055,
                )
            )

        output_arrows = VGroup()

        for method in methods:

            output_arrows.add(
                Arrow(
                    method.get_right(),
                    mask.get_left(),
                    buff=0.12,
                    stroke_width=1.7,
                    max_tip_length_to_length_ratio=0.055,
                )
            )

        self.play(
            FadeIn(source),
            run_time=0.4,
        )

        self.play(
            *[
                GrowArrow(a)
                for a in input_arrows
            ],
            *[
                FadeIn(m)
                for m in methods
            ],
            run_time=0.7,
        )

        self.play(
            *[
                GrowArrow(a)
                for a in output_arrows
            ],
            FadeIn(mask),
            run_time=0.7,
        )

        mask_label = Text(
            "TAMPER MASK",
            font_size=18,
            color=self.RED_C,
            weight=BOLD,
        )

        mask_label.next_to(
            mask,
            DOWN,
            buff=0.18,
        )

        self.play(
            Write(mask_label),
            run_time=0.4,
        )

        fusion_text = Text(
            "Deterministic Signal Fusion",
            font_size=21,
            color=self.YELLOW_C,
        )

        fusion_text.move_to([
            self.CONTENT_X,
            -1.35,
            0,
        ])

        question_text = Text(
            "Where does the image look manipulated?",
            font_size=18,
        )

        question_text.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(
            Write(fusion_text),
            run_time=0.5,
        )

        self.play(
            Write(question_text),
            run_time=0.5,
        )

        self.wait(0.8)

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

        self.play(
            FadeOut(scene5),
            run_time=0.6,
        )

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
            run_time=0.7,
        )

        mask2 = self.create_mask()

        vlm = self.node(
            "VISION\nLANGUAGE\nMODEL",
            2.0,
            1.3,
            self.TEAL_C,
            16,
        )

        semantic = self.node(
            "SEMANTIC\nREVIEW",
            1.9,
            1.15,
            self.GREEN_C,
            16,
        )

        vlm_flow = VGroup(
            mask2,
            vlm,
            semantic,
        )

        vlm_flow.arrange(
            RIGHT,
            buff=1.2,
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

        self.play(
            FadeIn(mask2),
            run_time=0.4,
        )

        self.play(
            GrowArrow(vlm_arrow1),
            FadeIn(vlm),
            run_time=0.5,
        )

        self.play(
            GrowArrow(vlm_arrow2),
            FadeIn(semantic),
            run_time=0.5,
        )

        semantic_items = VGroup(
            Text(
                "Impossible lighting",
                font_size=18,
            ),
            Text(
                "Anatomical inconsistencies",
                font_size=18,
            ),
            Text(
                "Texture anomalies",
                font_size=18,
            ),
        )

        semantic_items.arrange(
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
                run_time=0.3,
            )

        caption6 = Text(
            "The math tells us where. The VLM tells us why.",
            font_size=19,
            color=self.TEAL_C,
        )

        caption6.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(
            Write(caption6),
            run_time=0.6,
        )

        self.wait(0.8)

        scene6 = VGroup(
            scene6_title,
            vlm_flow,
            vlm_arrow1,
            vlm_arrow2,
            semantic_items,
            caption6,
        )

        self.play(
            FadeOut(scene6),
            run_time=0.6,
        )

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
            run_time=0.7,
        )

        conv_image = self.node(
            "IMAGE",
            1.3,
            0.9,
            self.BLUE_C,
            16,
        )

        feature_node = self.node(
            "FEATURE\nEXTRACTION",
            1.8,
            1.1,
            self.PURPLE_C,
            15,
        )

        convnext = self.node(
            "ConvNeXt",
            1.7,
            1.1,
            self.PURPLE_C,
            17,
        )

        score = self.node(
            "AI\n0.94",
            1.3,
            1.1,
            self.RED_C,
            17,
        )

        conv_nodes = VGroup(
            conv_image,
            feature_node,
            convnext,
            score,
        )

        conv_nodes.arrange(
            RIGHT,
            buff=0.85,
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

        self.play(
            FadeIn(conv_image),
            run_time=0.4,
        )

        for i in range(3):

            self.play(
                GrowArrow(conv_arrows[i]),
                FadeIn(conv_nodes[i + 1]),
                run_time=0.45,
            )

        latent = Text(
            "Subtle learned patterns associated with synthetic media",
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
            font_size=17,
        )

        generators.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(
            Write(latent),
            run_time=0.6,
        )

        self.play(
            Write(generators),
            run_time=0.5,
        )

        self.wait(0.8)

        scene7 = VGroup(
            scene7_title,
            conv_nodes,
            conv_arrows,
            latent,
            generators,
        )

        self.play(
            FadeOut(scene7),
            run_time=0.6,
        )

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
            run_time=0.7,
        )

        grad_image = self.node(
            "IMAGE",
            1.4,
            0.95,
            self.BLUE_C,
            17,
        )

        grad_network = self.node(
            "ConvNeXt",
            1.8,
            1.1,
            self.PURPLE_C,
            17,
        )

        heatmap = self.create_heatmap()

        grad_nodes = VGroup(
            grad_image,
            grad_network,
            heatmap,
        )

        grad_nodes.arrange(
            RIGHT,
            buff=1.3,
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

        self.play(
            FadeIn(grad_image),
            run_time=0.4,
        )

        self.play(
            GrowArrow(grad_arrow1),
            FadeIn(grad_network),
            run_time=0.5,
        )

        self.play(
            GrowArrow(grad_arrow2),
            FadeIn(heatmap),
            run_time=0.5,
        )

        gradient_label = Text(
            "Gradient-weighted activation",
            font_size=20,
            color=self.RED_C,
        )

        gradient_label.move_to([
            self.CONTENT_X,
            -1.25,
            0,
        ])

        grad_caption = Text(
            "Which regions drove the AI classification?",
            font_size=18,
        )

        grad_caption.move_to([
            self.CONTENT_X,
            self.CAPTION_Y,
            0,
        ])

        self.play(
            Write(gradient_label),
            run_time=0.5,
        )

        self.play(
            Write(grad_caption),
            run_time=0.5,
        )

        self.wait(0.8)

        scene8 = VGroup(
            scene8_title,
            grad_nodes,
            grad_arrow1,
            grad_arrow2,
            gradient_label,
            grad_caption,
        )

        self.play(
            FadeOut(scene8),
            run_time=0.6,
        )

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
            run_time=0.7,
        )

        provenance = self.node(
            "PROVENANCE\nC2PA",
            1.8,
            1.0,
            self.GREEN_C,
            15,
        )

        forensic = self.node(
            "FORENSICS\n+ VLM",
            1.8,
            1.0,
            self.YELLOW_C,
            15,
        )

        conv_score = self.node(
            "ConvNeXt\n0.94",
            1.8,
            1.0,
            self.PURPLE_C,
            15,
        )

        scores = VGroup(
            provenance,
            forensic,
            conv_score,
        )

        scores.arrange(
            RIGHT,
            buff=1.2,
        )

        scores.move_to([
            self.CONTENT_X,
            1.0,
            0,
        ])

        self.play(
            FadeIn(provenance),
            FadeIn(forensic),
            FadeIn(conv_score),
            run_time=0.6,
        )

        fusion_node = self.node(
            "CALIBRATED\nFUSION",
            2.0,
            1.05,
            self.TEAL_C,
            16,
        )

        fusion_node.move_to([
            self.CONTENT_X,
            -0.75,
            0,
        ])

        # --------------------------------------------------------
        # CLEAN LATE-FUSION ARROWS
        # Separate landing points and small arrowheads.
        # --------------------------------------------------------

        fusion_arrows = VGroup()

        target_points = [
            fusion_node.get_top() + LEFT * 0.60,
            fusion_node.get_top(),
            fusion_node.get_top() + RIGHT * 0.60,
        ]

        for score_node, target in zip(
            scores,
            target_points,
        ):

            arrow = Arrow(
                start=score_node.get_bottom(),
                end=target,
                buff=0.14,
                stroke_width=1.4,
                max_tip_length_to_length_ratio=0.045,
                max_stroke_width_to_length_ratio=3,
            )

            fusion_arrows.add(arrow)

        self.play(
            *[
                GrowArrow(a)
                for a in fusion_arrows
            ],
            FadeIn(fusion_node),
            run_time=0.8,
        )

        result = Text(
            "AI GENERATED  •  94% CONFIDENCE",
            font_size=22,
            color=self.RED_C,
            weight=BOLD,
        )

        result.move_to([
            self.CONTENT_X,
            -1.90,
            0,
        ])

        self.play(
            Write(result),
            run_time=0.6,
        )

        fusion_caption = Text(
            "Multiple independent signals. One calibrated decision.",
            font_size=18,
        )

        fusion_caption.move_to([
            self.CONTENT_X,
            -2.65,
            0,
        ])

        self.play(
            Write(fusion_caption),
            run_time=0.6,
        )

        # Let final fusion result breathe.
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
        # SLOW EXIT FROM FOCUS MODE
        # ========================================================

        # Remove foreground first.
        self.play(
            FadeOut(scene9),
            run_time=1.2,
        )

        # Dimmed pipeline remains visible.
        self.wait(0.8)

        # Slowly reveal full pipeline.
        self.play(
            FadeOut(self.dim_overlay),
            run_time=1.3,
        )

        # Let viewer see complete architecture again.
        self.wait(2.0)

        # Close architecture section.
        self.play(
            FadeOut(pipeline),
            run_time=1.2,
        )

        # Breathing room before product.
        self.wait(0.8)

        # ========================================================
        # SCENE 10 — PRODUCT DEMO 1
        # ========================================================

        demo1 = self.demo_screen(
            "PRODUCT DEMO 1",
            "Upload + Provenance Check",
            "Insert demo_1.mp4 here in post",
        )

        demo1.move_to(ORIGIN)

        self.play(
            FadeIn(
                demo1,
                shift=RIGHT * 0.15,
            ),
            run_time=1.0,
        )

        # 10-second product demo
        self.wait(10)

        # ========================================================
        # SCENE 11 — PRODUCT DEMO 2
        # ========================================================

        demo2 = self.demo_screen(
            "PRODUCT DEMO 2",
            "Forensics + Semantic Analysis",
            "Insert demo_2.mp4 here in post",
        )

        demo2.move_to(ORIGIN)

        self.play(
            ReplacementTransform(
                demo1,
                demo2,
            ),
            run_time=0.8,
        )

        # 10-second product demo
        self.wait(10)

        # ========================================================
        # SCENE 12 — PRODUCT DEMO 3
        # ========================================================

        demo3 = self.demo_screen(
            "PRODUCT DEMO 3",
            "AI Detection + Explainability",
            "Insert demo_3.mp4 here in post",
        )

        demo3.move_to(ORIGIN)

        self.play(
            ReplacementTransform(
                demo2,
                demo3,
            ),
            run_time=0.8,
        )

        # 10-second product demo
        self.wait(10)

        self.play(
            FadeOut(demo3),
            run_time=0.7,
        )

        # ========================================================
        # SCENE 13 — RESULT DASHBOARD
        # ========================================================

        result_title = Text(
            "TrueSight Result",
            font_size=38,
            color=self.TEAL_C,
            weight=BOLD,
        )

        result_title.move_to([
            self.CONTENT_X,
            self.TITLE_Y,
            0,
        ])

        self.play(
            Write(result_title),
            run_time=0.7,
        )

        verdict_box = RoundedRectangle(
            width=6.5,
            height=0.95,
            corner_radius=0.10,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        verdict_text = Text(
            "AI GENERATED   •   94% CONFIDENCE",
            font_size=24,
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
            1.55,
            0,
        ])

        self.play(
            Create(verdict_box),
            Write(verdict_text),
            run_time=0.7,
        )

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

            cards.add(
                self.result_card(
                    heading,
                    value,
                )
            )

        cards.arrange_in_grid(
            rows=2,
            cols=3,
            buff=(0.4, 0.35),
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
                run_time=0.2,
            )

        self.wait(2)

        # ========================================================
        # SCENE 14 — END
        # ========================================================

        self.play(
            FadeOut(cards),
            FadeOut(verdict_group),
            FadeOut(result_title),
            run_time=0.7,
        )

        final_title = Text(
            "TrueSight",
            font_size=64,
            color=self.TEAL_C,
            weight=BOLD,
        )

        final_subtitle = Text(
            "Know what's real.",
            font_size=30,
        )

        final_group = VGroup(
            final_title,
            final_subtitle,
        )

        final_group.arrange(
            DOWN,
            buff=0.25,
        )

        final_group.move_to(ORIGIN)

        self.play(
            Write(final_title),
            run_time=0.8,
        )

        self.play(
            Write(final_subtitle),
            run_time=0.6,
        )

        self.wait(2)

        self.play(
            FadeOut(final_group),
            run_time=1.0,
        )

    # ============================================================
    # HELPER — CONSISTENT TITLE SYSTEM
    # ============================================================

    def scene_title(
        self,
        title,
        subtitle,
        accent,
    ):

        title_obj = Text(
            title,
            font_size=35,
            weight=BOLD,
        )

        subtitle_obj = Text(
            subtitle,
            font_size=16,
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
    # HELPER — FULL-SCREEN STRONG DIM
    # ============================================================

    def dim_background(
        self,
        opacity=0.94,
    ):

        # Full-screen overlay.
        # NO BLUR.

        self.dim_overlay = Rectangle(
            width=config.frame_width,
            height=config.frame_height,
            stroke_width=0,
            fill_color=self.BG,
            fill_opacity=opacity,
        )

        self.dim_overlay.move_to(ORIGIN)

        self.play(
            FadeIn(self.dim_overlay),
            run_time=0.35,
        )

    # ============================================================
    # HELPER — STANDARD NODE
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
    # HELPER — CLEAN HORIZONTAL ARROW
    # ============================================================

    def horizontal_arrow(
        self,
        first,
        second,
    ):

        return Arrow(
            first.get_right(),
            second.get_left(),
            buff=0.12,
            stroke_width=1.7,
            max_tip_length_to_length_ratio=0.055,
            max_stroke_width_to_length_ratio=3,
        )

    # ============================================================
    # HELPER — FORENSIC MASK
    # ============================================================

    def create_mask(self):

        box = Square(
            side_length=1.5,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        pixels = VGroup()

        positions = [
            (-0.38, 0.32),
            (0.02, 0.40),
            (0.38, 0.12),
            (-0.22, -0.05),
            (0.15, -0.28),
            (0.40, -0.36),
        ]

        for x, y in positions:

            pixel = Square(
                side_length=0.22,
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
            side_length=1.5,
            stroke_color=self.RED_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.95,
        )

        heat = VGroup()

        for radius, opacity in [
            (0.55, 0.12),
            (0.37, 0.27),
            (0.21, 0.55),
        ]:

            circle = Circle(
                radius=radius,
                stroke_width=0,
                fill_color=self.RED_C,
                fill_opacity=opacity,
            )

            circle.move_to(
                box.get_center()
                + RIGHT * 0.13
                + UP * 0.09
            )

            heat.add(circle)

        label = Text(
            "GRAD-CAM",
            font_size=14,
            color=self.RED_C,
            weight=BOLD,
        )

        label.next_to(
            box,
            DOWN,
            buff=0.14,
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
            width=10.5,
            height=5.8,
            corner_radius=0.15,
            stroke_color=self.TEAL_C,
            stroke_width=2,
            fill_color=self.PANEL,
            fill_opacity=0.25,
        )

        top_bar = Rectangle(
            width=10.5,
            height=0.5,
            stroke_width=0,
            fill_color=self.PANEL,
            fill_opacity=1,
        )

        top_bar.move_to(
            outer.get_top()
            + DOWN * 0.25
        )

        dots = VGroup(
            Dot(radius=0.055),
            Dot(radius=0.055),
            Dot(radius=0.055),
        )

        dots.arrange(
            RIGHT,
            buff=0.10,
        )

        dots.move_to(
            top_bar.get_left()
            + RIGHT * 0.45
        )

        heading = Text(
            title,
            font_size=28,
            color=self.TEAL_C,
            weight=BOLD,
        )

        heading.move_to(
            outer.get_center()
            + UP * 1.65
        )

        subtitle_text = Text(
            subtitle,
            font_size=20,
        )

        subtitle_text.next_to(
            heading,
            DOWN,
            buff=0.18,
        )

        video = RoundedRectangle(
            width=8.6,
            height=2.6,
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

        play.scale(0.22)
        play.rotate(-PI / 2)
        play.move_to(video)

        note = Text(
            instruction,
            font_size=13,
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
            width=3.3,
            height=1.0,
            corner_radius=0.08,
            stroke_color=self.MUTED,
            stroke_width=1.2,
            fill_color=self.PANEL,
            fill_opacity=0.9,
        )

        heading_text = Text(
            heading,
            font_size=13,
            color=self.TEAL_C,
            weight=BOLD,
        )

        value_text = Text(
            value,
            font_size=14,
        )

        if value_text.width > 2.8:

            value_text.scale_to_fit_width(
                2.8
            )

        text = VGroup(
            heading_text,
            value_text,
        )

        text.arrange(
            DOWN,
            buff=0.09,
        )

        text.move_to(box)

        return VGroup(
            box,
            text,
        )