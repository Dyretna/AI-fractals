import os
from typing import Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


class HardwareConfig:
    def __init__(
        self,
        force_cpu: bool = False,
        gpu_memory_limit: Optional[int] = None,
        allow_growth: bool = True,
        mixed_precision: bool = True,
    ):
        if os.getenv("FORCE_CPU", "").lower() in ("true", "1", "yes"):
            force_cpu = True

        env_mem_limit = os.getenv("GPU_MEMORY_LIMIT", "").strip()
        if env_mem_limit and env_mem_limit.isdigit():
            gpu_memory_limit = int(env_mem_limit)

        env_allow_growth = os.getenv("GPU_ALLOW_GROWTH", "").lower()
        if env_allow_growth in ("true", "1", "yes"):
            allow_growth = True
        elif env_allow_growth in ("false", "0", "no"):
            allow_growth = False

        env_mixed_precision = os.getenv("GPU_MIXED_PRECISION", "").lower()
        if env_mixed_precision in ("true", "1", "yes"):
            mixed_precision = True
        elif env_mixed_precision in ("false", "0", "no"):
            mixed_precision = False

        self.force_cpu = force_cpu
        self.gpu_memory_limit = gpu_memory_limit
        self.allow_growth = allow_growth
        self.mixed_precision = mixed_precision
        self.has_gpu = False
        self.gpu_devices = []
        self.cuda_version = None
        self.cudnn_version = None
        self._tf_configured = False
        self._tf = None

    def detect_hardware(self) -> Dict:
        info = {
            "cuda_available": False,
            "gpu_available": False,
            "gpu_count": 0,
            "cuda_version": None,
            "cudnn_version": None,
        }

        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info["cuda_available"] = True
                info["gpu_devices"] = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        info["gpu_devices"].append(
                            {"name": parts[0], "driver": parts[1], "memory": parts[2]}
                        )
                info["gpu_count"] = len(info["gpu_devices"])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            import tensorflow as tf

            self._tf = tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus and not self.force_cpu:
                info["gpu_available"] = True
                self.has_gpu = True
                self.gpu_devices = gpus

                try:
                    from tensorflow.python.platform import build_info

                    info["cuda_version"] = build_info.build_info.get(
                        "cuda_version", "unknown"
                    )
                    info["cudnn_version"] = build_info.build_info.get(
                        "cudnn_version", "unknown"
                    )
                    self.cuda_version = info["cuda_version"]
                    self.cudnn_version = info["cudnn_version"]
                except Exception:
                    pass
        except ImportError:
            pass

        return info

    def configure_tensorflow(self) -> bool:
        if self._tf_configured:
            return self.has_gpu

        if self._tf is None:
            try:
                import tensorflow as tf

                self._tf = tf
            except ImportError:
                return False

        tf = self._tf

        if self.force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self._tf_configured = True
            return False

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            self._tf_configured = True
            return False

        try:
            for gpu in gpus:
                if self.allow_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)

                if self.gpu_memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=self.gpu_memory_limit
                            )
                        ],
                    )

            if self.mixed_precision and self.has_gpu:
                try:
                    policy = tf.keras.mixed_precision.Policy("mixed_float16")
                    tf.keras.mixed_precision.set_global_policy(policy)
                except Exception:
                    pass

            self.has_gpu = True
            self._tf_configured = True
            return True

        except RuntimeError:
            self._tf_configured = True
            return False

    def get_device_strategy(self):
        if self._tf is None:
            import tensorflow as tf

            self._tf = tf

        if not self._tf_configured:
            self.configure_tensorflow()

        if self.has_gpu and len(self.gpu_devices) > 1:
            return self._tf.distribute.MirroredStrategy()
        else:
            return self._tf.distribute.get_strategy()

    def print_summary(self):
        info = self.detect_hardware()

        print("\n" + "=" * 60)
        print("HARDWARE CONFIGURATION SUMMARY")
        print("=" * 60)

        if info["cuda_available"]:
            print("[OK] CUDA: Available")
            for i, gpu in enumerate(info.get("gpu_devices", [])):
                print(f"  GPU {i}: {gpu['name']}")
                print(f"         Driver: {gpu['driver']}")
                print(f"         Memory: {gpu['memory']}")
        else:
            print("[NO] CUDA: Not detected (nvidia-smi not found)")

        print()

        if self._tf and self._tf.test.is_built_with_cuda():
            print("[OK] TensorFlow: Built with CUDA")
        else:
            print("[NO] TensorFlow: Not built with CUDA")

        if info["gpu_available"]:
            print(f"[OK] TensorFlow GPU: {len(self.gpu_devices)} device(s) detected")
            if info["cuda_version"]:
                print(f"  CUDA Version: {info['cuda_version']}")
            if info["cudnn_version"]:
                print(f"  cuDNN Version: {info['cudnn_version']}")
        else:
            print("[NO] TensorFlow GPU: Not available")

        print()
        print("CONFIGURATION:")
        print(f"  Mode: {'CPU' if not self.has_gpu else 'GPU'}")
        if self.has_gpu:
            print(f"  Memory growth: {self.allow_growth}")
            if self.gpu_memory_limit:
                print(f"  Memory limit: {self.gpu_memory_limit} MB")
            print(f"  Mixed precision: {self.mixed_precision}")

        print("=" * 60 + "\n")


_global_config: Optional[HardwareConfig] = None


def setup_hardware(
    force_cpu: bool = False,
    gpu_memory_limit: Optional[int] = None,
    allow_growth: bool = True,
    mixed_precision: bool = True,
    verbose: bool = True,
) -> HardwareConfig:
    global _global_config

    _global_config = HardwareConfig(
        force_cpu=force_cpu,
        gpu_memory_limit=gpu_memory_limit,
        allow_growth=allow_growth,
        mixed_precision=mixed_precision,
    )

    _global_config.detect_hardware()
    _global_config.configure_tensorflow()

    if verbose:
        _global_config.print_summary()

    return _global_config


def get_hardware_config() -> HardwareConfig:
    global _global_config
    if _global_config is None:
        _global_config = setup_hardware(verbose=False)
    return _global_config


def check_gpu_available() -> bool:
    config = get_hardware_config()
    return config.has_gpu


def get_optimal_batch_size(base_size: int = 32, image_size: tuple = (256, 256)) -> int:
    config = get_hardware_config()

    if not config.has_gpu:
        return max(4, base_size // 8)

    if config.gpu_memory_limit:
        mem_gb = config.gpu_memory_limit / 1024
    else:
        mem_gb = 6

    pixels = image_size[0] * image_size[1]
    if pixels > 512 * 512:
        return max(8, int(base_size * (mem_gb / 8) * 0.5))
    else:
        return max(16, int(base_size * (mem_gb / 8)))
