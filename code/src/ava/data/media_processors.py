"""
Media processors for audio and video data in multi-modal training.

This module provides production-ready audio and video processing capabilities:
- AudioProcessor: Load, resample, and convert audio to mel spectrograms or waveforms
- VideoProcessor: Extract frames, resize, and normalize video data

Dependencies:
- Audio: torchaudio (pip install torchaudio)
- Video: torchvision with video backend (pip install torchvision)

Example:
    >>> audio_processor = AudioProcessor(sample_rate=16000, n_mels=80)
    >>> mel_spec = audio_processor.process("/path/to/audio.wav")

    >>> video_processor = VideoProcessor(num_frames=8, frame_size=(224, 224))
    >>> frames = video_processor.process("/path/to/video.mp4")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check for torchaudio availability
try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.debug("torchaudio not available - audio processing will use fallback")

# Check for torchvision availability
try:
    import torchvision
    import torchvision.transforms as VT
    from torchvision.io import read_video, read_image
    TORCHVISION_AVAILABLE = True

    # Check for video backend
    try:
        from torchvision.io import VideoReader
        VIDEO_READER_AVAILABLE = True
    except ImportError:
        VIDEO_READER_AVAILABLE = False
except ImportError:
    TORCHVISION_AVAILABLE = False
    VIDEO_READER_AVAILABLE = False
    logger.debug("torchvision not available - video processing will use fallback")


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    power: float = 2.0
    normalized: bool = True
    center: bool = True
    pad_mode: str = "reflect"
    norm: Optional[str] = "slaney"
    mel_scale: str = "htk"
    max_duration: Optional[float] = None  # Max duration in seconds (None = no limit)
    output_type: str = "mel_spectrogram"  # "mel_spectrogram", "waveform", "mfcc"


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    num_frames: int = 8
    frame_size: Tuple[int, int] = (224, 224)
    fps: Optional[float] = None  # Target FPS (None = use uniform sampling)
    channels: int = 3
    normalize: bool = True
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    output_format: str = "CTHW"  # "CTHW" or "TCHW"
    sampling_strategy: str = "uniform"  # "uniform", "random", "first"
    resize_mode: str = "bilinear"  # "bilinear", "bicubic", "nearest"


class AudioProcessor:
    """
    Audio processor for loading and transforming audio files.

    Supports:
    - Loading from file paths or raw tensors
    - Resampling to target sample rate
    - Conversion to mel spectrogram, waveform, or MFCC
    - Normalization and padding

    Args:
        config: AudioConfig with processing parameters
        device: Target device for output tensors

    Example:
        >>> processor = AudioProcessor(AudioConfig(sample_rate=16000, n_mels=80))
        >>> mel_spec = processor.process("audio.wav")
        >>> print(mel_spec.shape)  # [n_mels, time_frames]
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or AudioConfig()
        self.device = device or torch.device("cpu")
        self._transforms_initialized = False
        self._mel_transform = None
        self._mfcc_transform = None
        self._resamplers: Dict[int, Any] = {}

        if TORCHAUDIO_AVAILABLE:
            self._init_transforms()
        else:
            logger.warning(
                "torchaudio not available. Install with: pip install torchaudio. "
                "Using fallback implementation (returns zeros)."
            )

    def _init_transforms(self):
        """Initialize audio transforms."""
        if not TORCHAUDIO_AVAILABLE:
            return

        self._mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels,
            power=self.config.power,
            normalized=self.config.normalized,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            norm=self.config.norm,
            mel_scale=self.config.mel_scale,
        ).to(self.device)

        self._mfcc_transform = T.MFCC(
            sample_rate=self.config.sample_rate,
            n_mfcc=40,
            melkwargs={
                "n_fft": self.config.n_fft,
                "n_mels": self.config.n_mels,
                "hop_length": self.config.hop_length,
            },
        ).to(self.device)

        self._transforms_initialized = True

    def _get_resampler(self, orig_sr: int) -> Any:
        """Get or create a resampler for the given sample rate."""
        if not TORCHAUDIO_AVAILABLE:
            return None

        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = T.Resample(
                orig_freq=orig_sr,
                new_freq=self.config.sample_rate,
            ).to(self.device)
        return self._resamplers[orig_sr]

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform [channels, samples], sample_rate)
        """
        if not TORCHAUDIO_AVAILABLE:
            # Return dummy data
            samples = int(self.config.sample_rate * 1.0)
            return torch.zeros(1, samples), self.config.sample_rate

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(str(audio_path))
        return waveform, sample_rate

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Audio tensor [channels, samples]
            orig_sr: Original sample rate

        Returns:
            Resampled audio tensor
        """
        if orig_sr == self.config.sample_rate:
            return waveform

        if not TORCHAUDIO_AVAILABLE:
            # Approximate resampling via interpolation
            ratio = self.config.sample_rate / orig_sr
            new_length = int(waveform.shape[-1] * ratio)
            return F.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        resampler = self._get_resampler(orig_sr)
        return resampler(waveform)

    def to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: Audio tensor [channels, samples]

        Returns:
            Mel spectrogram [n_mels, time_frames]
        """
        if not TORCHAUDIO_AVAILABLE or self._mel_transform is None:
            # Return dummy spectrogram
            time_frames = waveform.shape[-1] // self.config.hop_length + 1
            return torch.zeros(self.config.n_mels, time_frames)

        waveform = waveform.to(self.device)
        mel_spec = self._mel_transform(waveform)

        # Convert to log scale (dB)
        mel_spec = torch.log(mel_spec + 1e-9)

        # If stereo, average channels
        if mel_spec.dim() == 3 and mel_spec.shape[0] > 1:
            mel_spec = mel_spec.mean(dim=0)
        elif mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)

        return mel_spec

    def to_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to MFCC features.

        Args:
            waveform: Audio tensor [channels, samples]

        Returns:
            MFCC features [n_mfcc, time_frames]
        """
        if not TORCHAUDIO_AVAILABLE or self._mfcc_transform is None:
            time_frames = waveform.shape[-1] // self.config.hop_length + 1
            return torch.zeros(40, time_frames)

        waveform = waveform.to(self.device)
        mfcc = self._mfcc_transform(waveform)

        if mfcc.dim() == 3 and mfcc.shape[0] > 1:
            mfcc = mfcc.mean(dim=0)
        elif mfcc.dim() == 3:
            mfcc = mfcc.squeeze(0)

        return mfcc

    def process(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Process audio input to the configured output format.

        Args:
            audio_input: File path or audio tensor
            sample_rate: Sample rate (required if audio_input is a tensor)

        Returns:
            Processed audio tensor (format depends on config.output_type)
        """
        # Load audio if path
        if isinstance(audio_input, (str, Path)):
            waveform, orig_sr = self.load_audio(audio_input)
        else:
            waveform = audio_input
            orig_sr = sample_rate or self.config.sample_rate

        # Resample if needed
        if orig_sr != self.config.sample_rate:
            waveform = self.resample(waveform, orig_sr)

        # Apply max duration limit
        if self.config.max_duration is not None:
            max_samples = int(self.config.max_duration * self.config.sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]

        # Convert to target format
        if self.config.output_type == "waveform":
            return waveform
        elif self.config.output_type == "mel_spectrogram":
            return self.to_mel_spectrogram(waveform)
        elif self.config.output_type == "mfcc":
            return self.to_mfcc(waveform)
        else:
            raise ValueError(f"Unknown output_type: {self.config.output_type}")


class VideoProcessor:
    """
    Video processor for loading and transforming video files.

    Supports:
    - Loading from file paths or tensors
    - Frame extraction with various sampling strategies
    - Resizing and normalization
    - Output format conversion (CTHW or TCHW)

    Args:
        config: VideoConfig with processing parameters
        device: Target device for output tensors

    Example:
        >>> processor = VideoProcessor(VideoConfig(num_frames=8, frame_size=(224, 224)))
        >>> frames = processor.process("video.mp4")
        >>> print(frames.shape)  # [3, 8, 224, 224] for CTHW format
    """

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or VideoConfig()
        self.device = device or torch.device("cpu")

        if not TORCHVISION_AVAILABLE:
            logger.warning(
                "torchvision not available. Install with: pip install torchvision. "
                "Using fallback implementation (returns zeros)."
            )

        self._init_transforms()

    def _init_transforms(self):
        """Initialize video transforms."""
        if not TORCHVISION_AVAILABLE:
            self._resize = None
            self._normalize = None
            return

        self._resize = VT.Resize(
            self.config.frame_size,
            interpolation=VT.InterpolationMode.BILINEAR
            if self.config.resize_mode == "bilinear"
            else VT.InterpolationMode.BICUBIC
            if self.config.resize_mode == "bicubic"
            else VT.InterpolationMode.NEAREST,
            antialias=True,
        )

        if self.config.normalize:
            self._normalize = VT.Normalize(
                mean=self.config.mean,
                std=self.config.std,
            )
        else:
            self._normalize = None

    def load_video(
        self, video_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Load video from file.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (frames [T, H, W, C], fps, metadata)
        """
        if not TORCHVISION_AVAILABLE:
            # Return dummy frames
            return (
                torch.zeros(
                    self.config.num_frames,
                    self.config.frame_size[0],
                    self.config.frame_size[1],
                    self.config.channels,
                ),
                30.0,
                {},
            )

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Try VideoReader first (more efficient), fall back to read_video
        if VIDEO_READER_AVAILABLE:
            try:
                reader = VideoReader(str(video_path), "video")
                metadata = reader.get_metadata()
                fps = metadata.get("video", {}).get("fps", [30.0])[0]

                frames = []
                for frame in reader:
                    frames.append(frame["data"])
                    if len(frames) >= self.config.num_frames * 2:  # Read extra for sampling
                        break

                if frames:
                    video_tensor = torch.stack(frames)  # [T, C, H, W]
                    video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, H, W, C]
                    return video_tensor, fps, metadata
            except Exception as e:
                logger.debug(f"VideoReader failed, falling back to read_video: {e}")

        # Fallback to read_video
        video_tensor, audio, info = read_video(str(video_path), pts_unit="sec")
        fps = info.get("video_fps", 30.0)

        return video_tensor, fps, info

    def sample_frames(
        self, video_tensor: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """
        Sample frames from video according to sampling strategy.

        Args:
            video_tensor: Video frames [T, H, W, C] or [T, C, H, W]
            num_frames: Number of frames to sample

        Returns:
            Sampled frames [num_frames, H, W, C] or [num_frames, C, H, W]
        """
        total_frames = video_tensor.shape[0]

        if total_frames <= num_frames:
            # Pad with last frame if not enough frames
            padding_needed = num_frames - total_frames
            if padding_needed > 0:
                last_frame = video_tensor[-1:].expand(padding_needed, -1, -1, -1)
                video_tensor = torch.cat([video_tensor, last_frame], dim=0)
            return video_tensor

        if self.config.sampling_strategy == "uniform":
            # Uniform sampling across the video
            indices = torch.linspace(0, total_frames - 1, num_frames).long()
        elif self.config.sampling_strategy == "random":
            # Random sampling
            indices = torch.randperm(total_frames)[:num_frames].sort()[0]
        elif self.config.sampling_strategy == "first":
            # Take first N frames
            indices = torch.arange(num_frames)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.config.sampling_strategy}")

        return video_tensor[indices]

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess frames: resize, normalize, and convert format.

        Args:
            frames: Video frames [T, H, W, C] or [T, C, H, W]

        Returns:
            Processed frames in configured output format
        """
        # Ensure TCHW format for processing
        if frames.dim() == 4 and frames.shape[-1] == self.config.channels:
            # THWC -> TCHW
            frames = frames.permute(0, 3, 1, 2)

        # Convert to float and scale to [0, 1]
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0

        # Resize each frame
        if TORCHVISION_AVAILABLE and self._resize is not None:
            resized_frames = []
            for i in range(frames.shape[0]):
                resized_frames.append(self._resize(frames[i]))
            frames = torch.stack(resized_frames)

        # Normalize
        if self._normalize is not None:
            normalized_frames = []
            for i in range(frames.shape[0]):
                normalized_frames.append(self._normalize(frames[i]))
            frames = torch.stack(normalized_frames)

        # Convert to output format
        if self.config.output_format == "CTHW":
            # TCHW -> CTHW
            frames = frames.permute(1, 0, 2, 3)
        # else: keep TCHW format

        return frames

    def process(
        self, video_input: Union[str, Path, torch.Tensor]
    ) -> torch.Tensor:
        """
        Process video input to the configured output format.

        Args:
            video_input: File path or video tensor

        Returns:
            Processed video tensor
        """
        # Load video if path
        if isinstance(video_input, (str, Path)):
            video_tensor, fps, metadata = self.load_video(video_input)
        else:
            video_tensor = video_input

        # Sample frames
        frames = self.sample_frames(video_tensor, self.config.num_frames)

        # Preprocess
        frames = self.preprocess_frames(frames)

        return frames.to(self.device)


# Factory functions for convenience
def create_audio_processor(
    sample_rate: int = 16000,
    n_mels: int = 80,
    output_type: str = "mel_spectrogram",
    **kwargs,
) -> AudioProcessor:
    """Create an AudioProcessor with common settings."""
    config = AudioConfig(
        sample_rate=sample_rate,
        n_mels=n_mels,
        output_type=output_type,
        **kwargs,
    )
    return AudioProcessor(config)


def create_video_processor(
    num_frames: int = 8,
    frame_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    **kwargs,
) -> VideoProcessor:
    """Create a VideoProcessor with common settings."""
    config = VideoConfig(
        num_frames=num_frames,
        frame_size=frame_size,
        normalize=normalize,
        **kwargs,
    )
    return VideoProcessor(config)


__all__ = [
    "AudioConfig",
    "VideoConfig",
    "AudioProcessor",
    "VideoProcessor",
    "create_audio_processor",
    "create_video_processor",
    "TORCHAUDIO_AVAILABLE",
    "TORCHVISION_AVAILABLE",
]
