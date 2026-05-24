"""
Automatic fractal generation pipeline.
Complete implementation using methods from Youvan (2024).
"""

import atexit
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import tensorflow as tf

    _tf_available = True
except ImportError:
    _tf_available = False

from ..generators import JuliaGenerator, MandelbrotGenerator
from .parameter_sampler import ParameterSampler
from .quality_evaluator import FractalQualityEvaluator


class AutomaticFractalPipeline:
    """
    Automated fractal generation pipeline using paper-based methods.

    From Youvan (2024) "AI-Enhanced Fractal Geometry":
    - Intelligent parameter exploration
    - Quality-based filtering
    - Automatic shoreline extraction
    - Comprehensive metadata tracking
    """

    def __init__(
        self,
        output_dir: str = "dataset",
        target_images: int = 1000,
        quality_threshold: float = 0.65,
        parallel_workers: int = 4,
        fractal_type: str = "mandelbrot",
    ):
        self.output_dir = Path(output_dir)
        self.target_images = target_images
        self.batch_size = max(4, parallel_workers)
        self.fractal_type = fractal_type
        self._interrupted = False

        # Create output directories
        self.fractal_dir = self.output_dir / "fractals" / fractal_type
        self.shoreline_dir = self.output_dir / "shorelines" / fractal_type
        self.metadata_dir = self.output_dir / "metadata" / fractal_type

        for directory in [self.fractal_dir, self.shoreline_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.evaluator = FractalQualityEvaluator(quality_threshold=quality_threshold)
        self.sampler = ParameterSampler(fractal_type=fractal_type)

        # Statistics tracking
        self.stats = {
            "generated": 0,
            "accepted": 0,
            "rejected": 0,
            "quality_scores": [],
            "fractal_dimensions": [],
            "acceptance_rate": 0.0,
            "start_time": None,
            "end_time": None,
        }

        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals (Ctrl+C, kill) for clean shutdown."""
        print("\n\n!!! Interrupt signal received. Cleaning up...")
        self._interrupted = True
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Cleanup resources before exit."""
        if _tf_available:
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

        if self.stats.get("start_time") and not self.stats.get("end_time"):
            self.stats["end_time"] = datetime.now()
            try:
                self.generate_report()
            except Exception:
                pass

    def generate_fractal(self, params: Dict) -> np.ndarray:
        """
        Generate fractal image with given parameters.

        Args:
            params: Parameter dictionary

        Returns:
            Generated fractal image
        """
        if self.fractal_type == "mandelbrot":
            gen = MandelbrotGenerator(
                width=params["width"],
                height=params["height"],
                max_iter=params["max_iter"],
            )
            img = gen.generate(
                xmin=params.get("x_min", params.get("xmin")),
                xmax=params.get("x_max", params.get("xmax")),
                ymin=params.get("y_min", params.get("ymin")),
                ymax=params.get("y_max", params.get("ymax")),
            )
        else:  # julia
            gen = JuliaGenerator(
                width=params["width"],
                height=params["height"],
                max_iter=params["max_iter"],
            )
            c = complex(params["c_real"], params["c_imag"])
            img = gen.generate(
                c=c,
                xmin=params.get("x_min", params.get("xmin")),
                xmax=params.get("x_max", params.get("xmax")),
                ymin=params.get("y_min", params.get("ymin")),
                ymax=params.get("y_max", params.get("ymax")),
            )

        return img

    def process_single(self, idx: int) -> Tuple[bool, Dict]:
        """
        Process single fractal generation + evaluation.

        Args:
            idx: Image index

        Returns:
            Tuple of (accepted, stats_dict)
        """
        try:
            # Sample parameters
            params = self.sampler.sample(strategy="mixed")

            # Generate fractal
            img = self.generate_fractal(params)

            # Evaluate quality
            score, accept, metrics = self.evaluator.evaluate(img)

            # Record statistics
            stats = {
                "score": score,
                "accept": accept,
                "fractal_dimension": metrics.get("fractal_dimension", 0.0),
            }

            if accept:
                # Record success for adaptive sampling
                self.sampler.record_success(params, score)

                # Extract shoreline
                shoreline = self.evaluator.extract_shoreline_canny(img)

                # Save images and metadata
                self.save_result(idx, img, shoreline, params, metrics)

            return accept, stats

        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            return False, {"score": 0.0, "accept": False, "fractal_dimension": 0.0}

    def save_result(
        self,
        idx: int,
        fractal: np.ndarray,
        shoreline: np.ndarray,
        params: Dict,
        metrics: Dict,
    ):
        """
        Save fractal, shoreline, and metadata.

        Args:
            idx: Image index
            fractal: Fractal image
            shoreline: Shoreline image
            params: Generation parameters
            metrics: Quality metrics
        """
        # Generate unique ID
        image_id = f"{self.fractal_type}_{idx:06d}"

        # Save fractal image
        fractal_path = self.fractal_dir / f"{image_id}.png"
        cv2.imwrite(str(fractal_path), fractal)

        # Save shoreline
        shoreline_path = self.shoreline_dir / f"{image_id}_shoreline.png"
        cv2.imwrite(str(shoreline_path), shoreline)

        # Create comprehensive metadata
        metadata = {
            "image_id": image_id,
            "fractal_type": self.fractal_type,
            "parameters": params,
            "quality_metrics": metrics,
            "paths": {
                "fractal": str(fractal_path),
                "shoreline": str(shoreline_path),
            },
            "timestamp": datetime.now().isoformat(),
            "paper_reference": "Youvan_2024_AI-Enhanced_Fractal_Geometry",
        }

        # Save metadata
        metadata_path = self.metadata_dir / f"{image_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def run(self):
        print(f"\n{'=' * 60}")
        print("Automatic Fractal Generation Pipeline")
        print("Based on: Youvan (2024) AI-Enhanced Fractal Geometry")
        print(f"{'=' * 60}")
        print(f"Fractal type: {self.fractal_type}")
        print(f"Target images: {self.target_images}")
        print(f"Quality threshold: {self.evaluator.quality_threshold}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 60}\n")

        self.stats["start_time"] = datetime.now()

        try:
            with tqdm(total=self.target_images, desc="Accepted images") as pbar:
                idx = 0
                while (
                    self.stats["accepted"] < self.target_images
                    and not self._interrupted
                ):
                    try:
                        accept, result_stats = self.process_single(idx)

                        self.stats["generated"] += 1
                        self.stats["quality_scores"].append(result_stats["score"])

                        if accept:
                            self.stats["accepted"] += 1
                            self.stats["fractal_dimensions"].append(
                                result_stats["fractal_dimension"]
                            )
                            pbar.update(1)
                        else:
                            self.stats["rejected"] += 1

                        self.stats["acceptance_rate"] = (
                            self.stats["accepted"] / self.stats["generated"]
                        )

                        pbar.set_postfix(
                            {
                                "accept_rate": f"{self.stats['acceptance_rate']:.1%}",
                                "avg_score": f"{np.mean(self.stats['quality_scores']):.3f}",
                            }
                        )

                        idx += 1

                    except KeyboardInterrupt:
                        self._interrupted = True
                        break
                    except Exception as e:
                        print(f"\nError processing image {idx}: {e}")
                        idx += 1
                        continue
        finally:
            self.stats["end_time"] = datetime.now()
            self.generate_report()

    def generate_report(self):
        """Generate comprehensive pipeline report."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        report = {
            "pipeline_configuration": {
                "fractal_type": self.fractal_type,
                "target_images": self.target_images,
                "quality_threshold": self.evaluator.quality_threshold,
                "batch_size": self.batch_size,
            },
            "statistics": {
                "total_generated": self.stats["generated"],
                "accepted": self.stats["accepted"],
                "rejected": self.stats["rejected"],
                "acceptance_rate": f"{self.stats['acceptance_rate']:.2%}",
                "duration_seconds": duration,
                "images_per_minute": self.stats["accepted"] / (duration / 60),
            },
            "quality_metrics": {
                "mean_score": float(np.mean(self.stats["quality_scores"])),
                "std_score": float(np.std(self.stats["quality_scores"])),
                "mean_fractal_dimension": float(
                    np.mean(self.stats["fractal_dimensions"])
                ),
                "std_fractal_dimension": float(
                    np.std(self.stats["fractal_dimensions"])
                ),
            },
            "timing": {
                "start_time": self.stats["start_time"].isoformat(),
                "end_time": self.stats["end_time"].isoformat(),
                "duration": str(self.stats["end_time"] - self.stats["start_time"]),
            },
        }

        # Save report
        report_path = self.output_dir / f"pipeline_report_{self.fractal_type}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'=' * 60}")
        print("Pipeline Complete!")
        print(f"{'=' * 60}")
        print(f"Generated: {self.stats['generated']}")
        print(f"Accepted: {self.stats['accepted']}")
        print(f"Rejected: {self.stats['rejected']}")
        print(f"Acceptance rate: {self.stats['acceptance_rate']:.1%}")
        print(f"Duration: {duration / 60:.1f} minutes")
        print(f"Speed: {self.stats['accepted'] / (duration / 60):.1f} images/minute")
        print(f"\nAverage quality score: {np.mean(self.stats['quality_scores']):.3f}")
        print(
            f"Average fractal dimension: {np.mean(self.stats['fractal_dimensions']):.3f}"
        )
        print(f"\nReport saved to: {report_path}")
        print(f"{'=' * 60}\n")

        return report
